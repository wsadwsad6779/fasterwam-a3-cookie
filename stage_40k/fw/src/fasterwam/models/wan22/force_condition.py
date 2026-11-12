"""Force-conditioning modules for FAWAM-style joint flow matching (P2 + P3).

Three pieces, all of which are added to the *action* expert only:

1. `ForceHistoryEncoder` -- a causal temporal encoder over the last `Hf` force
   observations ending at the current step. It is the carrier for P2.
2. `ForceFutureEncoder` -- embeds the (during training) *noised* future-force
   trajectory. Feeding the noisy target rather than the clean target is what
   makes the force head a velocity field of a real flow-matching model instead
   of a bolted-on MSE regressor.
3. `ForceTokenCrossAttention` -- per-token additive injection into the 32 action
   tokens, plus `ForceVelocityHead`, the per-token velocity predictor for the
   force stream.

Why additive cross-attention and not AdaLN-Zero: `SparseActionDiT.pre_dit`
builds `t_mod` as `time_projection(t).unflatten(1, (6, hidden_dim))`, which has
no time axis -- shape `[B, 6, D]` shared by every action token. A per-step force
history therefore cannot ride on the existing modulation path. Injecting into
the tokens themselves propagates through self-attention to all 30 blocks
(including the 22 context-free `ActionOnlyDiTBlock`s) without changing
`action_seq_len`, the RoPE cache, or the `video_kv_fusion` masks.

Step-0 equivalence: both the cross-attention output projection and the velocity
head are zero-initialised (`bias=False`), so a freshly built model is
numerically identical to the unmodified action expert on the action path.
Consequence worth remembering: a zero-initialised layer receives gradient itself
(`dL/dW = delta_out^T . h` with `h != 0`) but everything *behind* it receives
zero gradient on the first step. The first optimizer step therefore only moves
`force_inject.out_proj` and `force_head.linear.2`; the encoder becomes trainable
from step 1. The gradient probe checks step 1, not step 0.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class _CausalResidualBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm_mlp = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, key_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.norm_attn(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_mask, need_weights=False)[0]
        x = x + self.ffn(self.norm_mlp(x))
        return x


def _zero_final_linear(sequential: nn.Sequential) -> None:
    """Zero only the *last* layer of an MLP head.

    Zeroing every linear would be fatal, not merely conservative: with a zero
    hidden layer the output-gradient w.r.t. the last weight is exactly zero, so
    the head could never learn. Only the final projection may start at zero.
    """
    last = sequential[-1]
    if not isinstance(last, nn.Linear):
        raise TypeError(f"expected a Linear final layer, got {type(last)}")
    if last.bias is not None:
        raise ValueError("zero-initialised output projection must be created with bias=False")
    nn.init.zeros_(last.weight)


class ForceHistoryEncoder(nn.Module):
    """Causal encoder of the `Hf` force observations ending at the current step t.

    Window convention -- the dataset must match it exactly:
      * `force_history[:, i]` is the observation at absolute time
        `t - (hist_len - 1) + i`, so index `hist_len - 1` is always the current step.
      * When an episode has fewer than `Hf` past steps, the window is **left-padded**
        (invalid steps first) and `valid_mask` marks the trailing valid suffix.
      * The window ends at t, never at t+1, so nothing here can see the future.
    """

    def __init__(
        self,
        force_dim: int,
        hidden_dim: int = 256,
        hist_len: int = 16,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.0,
        scale: float = 1.0,
    ):
        super().__init__()
        if hist_len <= 0:
            raise ValueError(f"`hist_len` must be > 0, got {hist_len}")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` ({hidden_dim}) must be divisible by `num_heads` ({num_heads})"
            )
        self.force_dim = int(force_dim)
        self.hidden_dim = int(hidden_dim)
        self.hist_len = int(hist_len)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout = float(dropout)
        self.scale = float(scale)

        self.register_buffer("force_mean", torch.zeros(self.force_dim), persistent=True)
        self.register_buffer("force_std", torch.ones(self.force_dim), persistent=True)
        # 0.0 until `set_norm_stats` runs. It is a buffer so a checkpoint that
        # already carries calibration restores it, and a model that got neither
        # fails at the first forward instead of silently normalising by 1.0.
        self.register_buffer("stats_are_set", torch.zeros(()), persistent=True)

        self.value = nn.Linear(self.force_dim, self.hidden_dim)
        self.offset_emb = nn.Parameter(torch.zeros(self.hist_len, self.hidden_dim))
        self.blocks = nn.ModuleList(
            [
                _CausalResidualBlock(self.hidden_dim, self.num_heads, self.dropout)
                for _ in range(self.num_layers)
            ]
        )
        self.dropout_mod = nn.Dropout(self.dropout)

    def normalize(self, raw_force: torch.Tensor) -> torch.Tensor:
        """`(f - mean) / std * scale`, computed in float32 from the in-file buffers.

        The model owns force calibration: training windows and robot roll-outs
        both hand over raw newtons and both go through this function, so there
        is no second copy of the statistics that could drift (§四.6). The
        buffers travel inside `mot.state_dict()`, which makes a released
        checkpoint self-contained.
        """
        if float(self.stats_are_set.detach().float().item()) != 1.0:
            raise RuntimeError(
                "force calibration was never set: pass `force_norm_stats` when "
                "building the model, or load a checkpoint that contains it."
            )
        raw = raw_force.float()
        mean = self.force_mean.float()
        std = self.force_std.float().clamp(min=1e-8)
        return ((raw - mean) / std * self.scale).to(dtype=raw_force.dtype)

    def denormalize(self, normalized_force: torch.Tensor) -> torch.Tensor:
        """Inverse of `normalize`, for logging and for the P4 feedback hook."""
        if float(self.stats_are_set.detach().float().item()) != 1.0:
            raise RuntimeError("force calibration was never set")
        mean = self.force_mean.float()
        std = self.force_std.float().clamp(min=1e-8)
        return (normalized_force.float() / self.scale) * std + mean

    def set_norm_stats(self, mean, std, scale: float = 1.0) -> None:
        mean_t = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std_t = torch.as_tensor(std, dtype=torch.float32).reshape(-1)
        if mean_t.numel() != self.force_dim or std_t.numel() != self.force_dim:
            raise ValueError(
                f"force norm stats must have {self.force_dim} entries, "
                f"got mean={mean_t.numel()} std={std_t.numel()}"
            )
        if not torch.isfinite(std_t).all() or (std_t <= 0).any():
            raise ValueError(f"`force_std` must be finite and strictly positive, got {std_t.tolist()}")
        if not torch.isfinite(mean_t).all():
            raise ValueError("`force_mean` must be finite")
        self.force_mean.copy_(mean_t)
        self.force_std.copy_(std_t)
        self.scale = float(scale)
        self.stats_are_set.fill_(1.0)

    def forward(
        self,
        force: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if force.ndim != 3:
            raise ValueError(f"`force_history` must be 3D [B, H, F], got shape {tuple(force.shape)}")
        if force.shape[2] != self.force_dim:
            raise ValueError(
                f"`force_history` last dim must be {self.force_dim}, got {force.shape[2]}"
            )
        if force.shape[1] > self.hist_len:
            raise ValueError(
                f"force history length {force.shape[1]} exceeds configured hist_len {self.hist_len}"
            )
        if self.blocks and any(p.dtype != force.dtype for p in self.blocks[0].parameters()):
            raise RuntimeError("ForceHistoryEncoder called before `to(device, dtype)`.")

        # `offset_emb` is indexed by absolute distance from the current step, so a
        # short window takes the *tail* of the table -- row -1 always means "now".
        seq_len = force.shape[1]
        x = self.value(force) + self.dropout_mod(self.offset_emb[None, -seq_len:])

        key_mask = None
        if valid_mask is not None:
            if valid_mask.shape != force.shape[:2]:
                raise ValueError(
                    f"`force_history_mask` must be [B, H], got {tuple(valid_mask.shape)} "
                    f"vs force {tuple(force.shape)}"
                )
            valid_mask = valid_mask.to(device=force.device, dtype=torch.bool)
            n_valid = valid_mask.to(torch.int64).sum(dim=1)
            if not bool((n_valid > 0).all().item()):
                raise ValueError(
                    "each row needs at least one valid history step -- the current step itself; "
                    "a fully masked window leaves the force stream with no observation."
                )
            positions = torch.arange(seq_len, device=force.device, dtype=torch.int64)
            expected = positions[None, :] >= (seq_len - n_valid)[:, None]
            if not torch.equal(expected, valid_mask):
                raise ValueError(
                    "`force_history_mask` must mark the trailing valid suffix of every row "
                    "(left-padded windows only)."
                )
            key_mask = ~valid_mask

        for block in self.blocks:
            x = block(x, key_mask=key_mask)
        if key_mask is not None:
            # Masked attention already ignores the padding; zeroing the tokens too
            # keeps the kv block passed on to the cross-attention free of any value
            # produced from a padded position.
            x = torch.where(key_mask[:, :, None], torch.zeros_like(x), x)
        return x, key_mask


class ForceFutureEncoder(nn.Module):
    """Embeds the (noised) future-force trajectory aligned with the action tokens.

    `noisy_force[:, i]` is force step `t + 1 + i`, so token i of the action stream
    and force token i describe the same time step. A window that runs past the end
    of its episode is **right-padded** (valid steps first) and masked here, exactly
    like the padded tail of an action horizon.
    """

    def __init__(self, force_dim: int, hidden_dim: int = 256, horizon: int = 32, dropout: float = 0.0):
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"`horizon` must be > 0, got {horizon}")
        self.force_dim = int(force_dim)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.value = nn.Linear(self.force_dim, self.hidden_dim)
        self.step_emb = nn.Parameter(torch.zeros(self.horizon, self.hidden_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        noisy_force: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if noisy_force.ndim != 3:
            raise ValueError(
                f"`noisy_force_future` must be 3D [B, T, F], got shape {tuple(noisy_force.shape)}"
            )
        if noisy_force.shape[2] != self.force_dim:
            raise ValueError(
                f"`noisy_force_future` last dim must be {self.force_dim}, got {noisy_force.shape[2]}"
            )
        if noisy_force.shape[1] > self.horizon:
            raise ValueError(
                f"force horizon {noisy_force.shape[1]} exceeds configured horizon {self.horizon}"
            )
        seq_len = noisy_force.shape[1]
        x = self.dropout(self.value(noisy_force) + self.step_emb[None, :seq_len])

        key_mask = None
        if valid_mask is not None:
            if valid_mask.shape != noisy_force.shape[:2]:
                raise ValueError(
                    f"`force_future_mask` must be [B, T], got {tuple(valid_mask.shape)} "
                    f"vs force {tuple(noisy_force.shape)}"
                )
            valid_mask = valid_mask.to(device=noisy_force.device, dtype=torch.bool)
            n_valid = valid_mask.to(torch.int64).sum(dim=1)
            if not bool((n_valid > 0).all().item()):
                raise ValueError("every row needs at least one valid future force step")
            positions = torch.arange(seq_len, device=noisy_force.device, dtype=torch.int64)
            expected = positions[None, :] < n_valid[:, None]
            if not torch.equal(expected, valid_mask):
                raise ValueError(
                    "`force_future_mask` must mark the leading valid prefix of every row "
                    "(right-padded windows only)."
                )
            key_mask = ~valid_mask
            x = torch.where(key_mask[:, :, None], torch.zeros_like(x), x)
        return x, key_mask


class ForceTokenCrossAttention(nn.Module):
    """Action tokens as queries; encoded force tokens as keys/values."""

    def __init__(
        self,
        action_hidden_dim: int,
        force_hidden_dim: int,
        num_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        if action_hidden_dim % num_heads != 0:
            raise ValueError(
                f"`action_hidden_dim` ({action_hidden_dim}) must be divisible by "
                f"`num_heads` ({num_heads})"
            )
        self.action_hidden_dim = int(action_hidden_dim)
        self.force_hidden_dim = int(force_hidden_dim)
        self.num_heads = int(num_heads)
        self.attn = nn.MultiheadAttention(
            embed_dim=action_hidden_dim,
            num_heads=num_heads,
            kdim=force_hidden_dim,
            vdim=force_hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        # The single zero-init point for the whole injection path: out_proj is
        # the only layer between the force tokens and the action stream.
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)
        self.norm = nn.LayerNorm(action_hidden_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        force_tokens: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3 or force_tokens.ndim != 3:
            raise ValueError(
                f"`tokens` and `force_tokens` must be 3D, got {tuple(tokens.shape)} "
                f"and {tuple(force_tokens.shape)}"
            )
        if force_tokens.shape[2] != self.force_hidden_dim:
            raise ValueError(
                f"force token dim {force_tokens.shape[2]} != {self.force_hidden_dim}"
            )
        if key_mask is not None and not bool(key_mask.any().item()):
            key_mask = None
        h = self.norm(tokens)
        delta = self.attn(
            h,
            force_tokens,
            force_tokens,
            key_padding_mask=key_mask,
            need_weights=False,
        )[0]
        # A fully masked row makes MHA return NaN; those rows carry no force at
        # all, so the correct contribution is exactly zero.
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        return tokens + delta


class ForceVelocityHead(nn.Module):
    """Per-token velocity field for the force stream (same target form as action)."""

    def __init__(
        self,
        action_hidden_dim: int,
        force_dim: int,
        horizon: int = 32,
        hidden_dim: int = 512,
    ):
        super().__init__()
        if horizon <= 0:
            raise ValueError(f"`horizon` must be > 0, got {horizon}")
        self.action_hidden_dim = int(action_hidden_dim)
        self.force_dim = int(force_dim)
        self.horizon = int(horizon)
        self.step_emb = nn.Parameter(torch.zeros(self.horizon, self.action_hidden_dim))
        self.linear = nn.Sequential(
            nn.Linear(self.action_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.force_dim, bias=False),
        )
        _zero_final_linear(self.linear)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"action tokens must be 3D, got shape {tuple(tokens.shape)}")
        seq_len = tokens.shape[1]
        if seq_len > self.horizon:
            raise ValueError(
                f"action token length {seq_len} exceeds force horizon {self.horizon}; "
                "the head cannot address per-step targets."
            )
        return self.linear(tokens + self.step_emb[None, :seq_len])


def build_force_conditioner(cfg: Optional[Dict[str, Any]]):
    """Build the four force modules and freeze their shared geometry.

    The sub-configs are written independently in YAML, so nothing would stop a
    `force_dim` in the history encoder from disagreeing with the future encoder
    or the head. That mismatch would not raise until some forward deep inside
    attention, possibly after training started, so the geometry is checked here
    and published as `conditioner.spec` for the rest of the model to read.
    """
    if not cfg:
        return None
    if not bool(cfg.get("enabled", False)):
        return None

    history = ForceHistoryEncoder(**cfg["history_encoder"])
    future = ForceFutureEncoder(**cfg["future_encoder"])
    inject = ForceTokenCrossAttention(**cfg["inject"])
    head = ForceVelocityHead(**cfg["force_head"])

    dims = {
        "history_encoder": history.force_dim,
        "future_encoder": future.force_dim,
        "force_head": head.force_dim,
    }
    if len(set(dims.values())) != 1:
        raise ValueError(f"`force_dim` disagrees across the force modules: {dims}")
    if head.horizon != future.horizon:
        raise ValueError(
            f"`horizon` disagrees: future_encoder={future.horizon} force_head={head.horizon}"
        )
    if history.hidden_dim != future.hidden_dim or inject.force_hidden_dim != history.hidden_dim:
        raise ValueError(
            "history encoder, future encoder and the injection keys must share one hidden dim: "
            f"{history.hidden_dim}, {future.hidden_dim}, {inject.force_hidden_dim}"
        )
    if inject.action_hidden_dim != head.action_hidden_dim:
        raise ValueError("`action_hidden_dim` disagrees between the injection and the head.")

    conditioner = nn.ModuleDict(
        {
            "history_encoder": history,
            "future_encoder": future,
            "inject": inject,
            "force_head": head,
        }
    )
    channels = cfg.get("channels")
    conditioner.spec = {
        "force_dim": int(history.force_dim),
        "hist_len": int(history.hist_len),
        "horizon": int(future.horizon),
        "hidden_dim": int(history.hidden_dim),
        "inject_num_heads": int(inject.num_heads),
        "action_hidden_dim": int(inject.action_hidden_dim),
        "channels": None if channels is None else tuple(int(c) for c in channels),
        "channel_note": cfg.get("channel_note"),
    }
    if channels is not None and len(conditioner.spec["channels"]) != conditioner.spec["force_dim"]:
        raise ValueError(
            f"`channels` lists {len(conditioner.spec['channels'])} entries but the modules expect "
            f"{conditioner.spec['force_dim']} force dimensions"
        )
    return conditioner
