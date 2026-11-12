from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from fasterwam.light_state_v3 import force_calibration_fingerprint
from fasterwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        loss_lambda_force: float = 0.0,
        mot_action_video_attention_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot
        self.mot_action_video_attention_layers = self._normalize_mot_action_video_attention_layers(
            mot_action_video_attention_layers,
            num_layers=int(self.mot.num_layers),
        )
        if self.mot_action_video_attention_layers is None:
            self._mot_action_video_attention_layer_set = frozenset(range(int(self.mot.num_layers)))
            logger.info(
                "MoT Action->Video attention is enabled for all %d layers.",
                int(self.mot.num_layers),
            )
        else:
            self._mot_action_video_attention_layer_set = frozenset(self.mot_action_video_attention_layers)
            logger.info(
                "MoT Action->Video attention enabled at layers %s (disabled elsewhere).",
                list(self.mot_action_video_attention_layers),
            )

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.force_spec = None
        self.loss_lambda_force = 0.0
        conditioner = getattr(getattr(self, "action_expert", None), "force_conditioner", None)
        if conditioner is not None:
            self.force_spec = dict(conditioner.spec)
            self.loss_lambda_force = float(loss_lambda_force)
            if self.loss_lambda_force <= 0.0:
                raise ValueError(
                    "`force_conditioner` is built but `loss_lambda_force` = "
                    f"{self.loss_lambda_force}; the force stream would never be trained."
                )
            # Action and force are one joint flow field: they share `t`, so
            # `train_action_scheduler` / `infer_action_scheduler` serve both
            # streams and the coupled sampler cannot drift out of sync. Only the
            # noise realization is independent, which is what makes the two
            # velocity targets distinguishable.

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)

    @property
    def force_conditioner(self):
        return getattr(self.action_expert, "force_conditioner", None)

    def set_force_norm_stats(self, mean, std, scale: float = 1.0):
        conditioner = self.force_conditioner
        if conditioner is None:
            raise RuntimeError("`set_force_norm_stats` requires a model built with `force_config`.")
        conditioner["history_encoder"].set_norm_stats(mean, std, scale=scale)

    def _require_force_training_inputs(self, sample):
        if self.force_spec is None:
            stray = [
                key for key in sample
                if key.startswith("force_") or key in ("force", "window_offset")
            ]
            if stray:
                raise ValueError(
                    "the sample carries force tensors but this model was built without "
                    f"`force_config`, so they would be dropped silently: {stray}"
                )
            return None
        for key in ("force_history", "force_history_mask", "force_future", "force_future_mask"):
            if key not in sample:
                raise ValueError(
                    f"this model was built with force conditioning; `sample['{key}']` is required"
                )
        spec = self.force_spec
        force_history = sample["force_history"]
        history_mask = sample["force_history_mask"]
        force_future = sample["force_future"]
        future_mask = sample["force_future_mask"]
        for name, tensor in (
            ("force_history", force_history),
            ("force_future", force_future),
        ):
            if tensor.ndim != 3 or tensor.shape[2] != spec["force_dim"]:
                raise ValueError(
                    f"`sample['{name}']` must be [B, T, {spec['force_dim']}], got shape "
                    f"{tuple(tensor.shape)}"
                )
        if force_history.shape[1] != spec["hist_len"]:
            raise ValueError(
                f"`force_history` must carry exactly the frozen hist_len {spec['hist_len']} steps "
                f"(left-padded at episode start), got {force_history.shape[1]}"
            )
        if force_future.shape[1] != spec["horizon"]:
            raise ValueError(
                f"`force_future` must carry exactly the frozen horizon {spec['horizon']} steps "
                f"(right-padded at episode end), got {force_future.shape[1]}"
            )
        for name, mask, data in (
            ("force_history_mask", history_mask, force_history),
            ("force_future_mask", future_mask, force_future),
        ):
            if mask.shape != data.shape[:2]:
                raise ValueError(
                    f"`sample['{name}']` must be {tuple(data.shape[:2])}, got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise ValueError(f"`sample['{name}']` must be a bool tensor")
        if not bool(history_mask.any(dim=1).all().item()):
            raise ValueError("a sample has an entirely masked force history window")
        if not bool(future_mask.any(dim=1).all().item()):
            raise ValueError("a sample has an entirely masked force future window")
        return force_history, history_mask, force_future, future_mask

    def _prepare_force_inputs(self, sample):
        """Move the force windows to the device and put them in model space.

        `normalize` is the only calibration step and it is applied here for
        training and in `_build_force_infer_state` for roll-outs, both calling
        the same method on the same buffers, so the two paths cannot diverge
        (§四.6). Downstream code never normalizes again.
        """
        prepared = self._require_force_training_inputs(sample)
        if prepared is None:
            return None
        force_history, history_mask, force_future, future_mask = prepared
        encoder = self.force_conditioner["history_encoder"]
        return {
            "history": encoder.normalize(
                force_history.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ),
            "history_mask": history_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            "future": encoder.normalize(
                force_future.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            ),
            "future_mask": future_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        }

    @torch.no_grad()
    def assert_future_frames_are_actually_future(
        self,
        force_future: torch.Tensor,
        future_mask: torch.Tensor,
        window_offset,
        retriever,
        atol: float = 1e-4,
    ) -> dict:
        """Data-side leak gate: raw `force_future[:, j]` must be the frame at +j+1.

        `window_offset[i]` is the dataset row of the window's current frame for
        sample i, so the absolute row of label j is `offset + j + 1`. `retriever(rows)`
        reads those rows straight out of the source array in physical units, bypassing
        every dataset transform, which is what makes this an independent check instead
        of a tautology over the same slice the dataset already took.

        Only the valid (unmasked) prefix is compared, because a label past the end of
        an episode has no row to reindex; the masked tail is instead required to be
        exactly zero, which is what the model was fed.
        """
        if self.force_spec is None:
            raise RuntimeError("the leak gate requires a model built with force conditioning")
        horizon = self.force_spec["horizon"]
        if force_future.ndim != 3 or force_future.shape[1] != horizon:
            raise ValueError(
                f"`force_future` must be [B, {horizon}, F], got {tuple(force_future.shape)}"
            )
        if future_mask.shape != force_future.shape[:2]:
            raise ValueError(
                f"`future_mask` must be {tuple(force_future.shape[:2])}, got {tuple(future_mask.shape)}"
            )
        offsets = torch.as_tensor(window_offset, dtype=torch.int64).reshape(-1)
        if offsets.numel() != force_future.shape[0]:
            raise ValueError(
                f"`window_offset` must have one entry per sample, got {offsets.numel()} "
                f"for batch {force_future.shape[0]}"
            )
        idx = offsets[:, None] + torch.arange(
            1, horizon + 1, device=force_future.device, dtype=torch.int64
        )
        truth = torch.as_tensor(
            retriever(idx.detach().cpu().numpy()),
            dtype=torch.float32,
            device=force_future.device,
        )
        if tuple(truth.shape) != tuple(force_future.shape):
            raise ValueError(
                f"reindexed truth {tuple(truth.shape)} != sample force_future {tuple(force_future.shape)}"
            )
        valid = future_mask.to(device=force_future.device, dtype=torch.bool)
        if not bool(valid.any(dim=1).all().item()):
            raise ValueError("a sample has an entirely masked force future window")
        pad = ~valid[:, :, None].expand_as(force_future)
        if bool(pad.any().item()) and float(force_future.detach()[pad].abs().max().item()) != 0.0:
            raise AssertionError(
                "the masked tail of `force_future` is not zero: the dataset would be training "
                "the force head on values it claims are absent."
            )
        observed = force_future.float()[valid].reshape(-1)
        expected = truth[valid].reshape(-1)
        if not bool(torch.isfinite(expected).all().item()):
            raise ValueError("reindexed truth contains NaN/inf; the source array is not clean.")
        max_abs_diff = float((expected - observed).abs().max().item())
        if max_abs_diff > atol:
            raise AssertionError(
                f"force_future is not the strictly-forward window: max abs diff {max_abs_diff} > {atol}"
            )
        return {
            "max_abs_diff": max_abs_diff,
            "horizon": int(horizon),
            "compared_steps": int(valid.sum().item()),
            "atol": float(atol),
        }

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        mot_action_video_attention_layers: Optional[Sequence[int]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            mot_action_video_attention_layers=mot_action_video_attention_layers,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        return self

    @staticmethod
    def _normalize_mot_action_video_attention_layers(
        layer_indices: Optional[Sequence[int]],
        num_layers: int,
    ) -> Optional[tuple[int, ...]]:
        if layer_indices is None:
            return None
        if isinstance(layer_indices, (str, bytes)):
            raise ValueError("`mot_action_video_attention_layers` must be a sequence of layer indices, not a string.")

        normalized = []
        seen = set()
        for raw_idx in layer_indices:
            if isinstance(raw_idx, bool) or not isinstance(raw_idx, int):
                raise ValueError(
                    "`mot_action_video_attention_layers` must contain integers in "
                    f"[0, {num_layers - 1}], got {raw_idx!r}."
                )
            if raw_idx < 0 or raw_idx >= num_layers:
                raise ValueError(
                    "`mot_action_video_attention_layers` indices out of range: "
                    f"got {raw_idx}, expected [0, {num_layers - 1}]."
                )
            if raw_idx in seen:
                raise ValueError(
                    f"`mot_action_video_attention_layers` contains duplicate layer index {raw_idx}."
                )
            seen.add(raw_idx)
            normalized.append(raw_idx)
        return tuple(sorted(normalized))

    @torch.no_grad()
    def _build_video_attention_mask(
        self,
        video_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "force": self._prepare_force_inputs(sample),
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self._build_video_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    @torch.no_grad()
    def _build_mot_attention_masks(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> list[torch.Tensor]:
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=action_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        return [attention_mask] * int(self.mot.num_layers)

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        force_inputs = inputs["force"]
        if force_inputs is None:
            force_kwargs = {}
            target_force = None
            force_future_mask = None
        else:
            # Joint flow matching over [action, force]: one shared probability
            # path parameter `t`, two independent noise realizations.
            force_future = force_inputs["future"]
            force_future_mask = force_inputs["future_mask"]
            noise_force = torch.randn_like(force_future)
            noisy_force = self.train_action_scheduler.add_noise(
                force_future, noise_force, timestep_action
            )
            target_force = self.train_action_scheduler.training_target(
                force_future, noise_force, timestep_action
            )
            force_kwargs = {
                "force_history": force_inputs["history"],
                "force_history_mask": force_inputs["history_mask"],
                "noisy_force_future": noisy_force,
                "force_future_mask": force_future_mask,
            }

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            **force_kwargs,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_masks = self._build_mot_attention_masks(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_masks,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        pred_force = None
        if self.force_conditioner is not None:
            # Second injection point, after the MoT layers, so the force
            # conditioning is not diluted by 30 rounds of self-attention before
            # it reaches the force head.
            force_tokens = self.action_expert.apply_force_injection_post_dit(
                tokens_out["action"], action_pre
            )
            pred_force = self.action_expert.predict_force_velocity(force_tokens, action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        loss_total = self.loss_lambda_video * loss_video + self.loss_lambda_action * loss_action
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }

        if pred_force is not None:
            force_loss_token = F.mse_loss(
                pred_force.float(), target_force.float(), reduction="none"
            )  # [B, T, F]
            # Masks mark the observed (leading) steps of a right-padded window --
            # the opposite polarity from the action loss's `action_is_pad`.
            valid_force = force_future_mask.to(dtype=force_loss_token.dtype)[:, :, None]
            force_steps = valid_force.sum(dim=(1, 2)).clamp(min=1.0) * pred_force.shape[-1]
            force_loss_per_sample = (force_loss_token * valid_force).sum(dim=(1, 2)) / force_steps
            force_weight = self.train_action_scheduler.training_weight(timestep_action).to(
                force_loss_per_sample.device, dtype=force_loss_per_sample.dtype
            )
            loss_force = (force_loss_per_sample * force_weight).mean()
            loss_total = loss_total + self.loss_lambda_force * loss_force
            loss_dict["loss_force"] = self.loss_lambda_force * float(loss_force.detach().item())
            loss_dict["force_weight_mean"] = float(force_weight.detach().float().mean().item())
            loss_dict["force_valid_step_frac"] = float(
                force_future_mask.float().mean().item()
            )
        return loss_total, loss_dict

    def _require_force_inference_inputs(self, force_history, force_history_mask):
        if self.force_conditioner is None:
            if force_history is not None:
                raise ValueError(
                    "`force_history` was provided but this model was built without `force_config`"
                )
            return False
        if force_history is None or force_history_mask is None:
            raise ValueError(
                "this model was trained with force conditioning, so `force_history` and "
                "`force_history_mask` are required at inference; passing None would feed the "
                "network an input distribution it never saw."
            )
        return True

    def _build_force_infer_state(
        self,
        force_history: Optional[torch.Tensor],
        force_history_mask: Optional[torch.Tensor],
        num_inference_steps: int,
        sigma_shift: Optional[float],
        seed: Optional[int],
        rand_device: str,
    ) -> Optional[dict]:
        if not self._require_force_inference_inputs(force_history, force_history_mask):
            return None
        cfg = self.force_spec
        if force_history.ndim == 2:
            force_history = force_history.unsqueeze(0)
        if force_history_mask.ndim == 1:
            force_history_mask = force_history_mask.unsqueeze(0)
        if force_history.ndim != 3 or force_history.shape[2] != cfg["force_dim"]:
            raise ValueError(
                f"`force_history` must be [B, H, {cfg['force_dim']}], got {tuple(force_history.shape)}"
            )
        if force_history.shape[1] != cfg["hist_len"]:
            raise ValueError(
                f"`force_history` must carry exactly the frozen hist_len {cfg['hist_len']} steps, "
                f"left-padded on the robot's first frames; got {force_history.shape[1]}"
            )
        if force_history_mask.shape != force_history.shape[:2]:
            raise ValueError(
                f"`force_history_mask` must be {tuple(force_history.shape[:2])}, "
                f"got {tuple(force_history_mask.shape)}"
            )
        force_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_force = torch.randn(
            (1, cfg["horizon"], cfg["force_dim"]),
            generator=force_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        # The force stream shares the action stream's schedule by construction,
        # so only the step sizes are needed to advance it.
        _, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_force.dtype,
            shift_override=sigma_shift,
        )
        encoder = self.force_conditioner["history_encoder"]
        history_norm = encoder.normalize(
            force_history.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        )
        return {
            "history": history_norm,
            "history_mask": force_history_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
            "latents": latents_force,
            # A roll-out has no episode end, so every sampled future step is a
            # real prediction target.
            "future_mask": torch.ones(
                latents_force.shape[:2], dtype=torch.bool, device=latents_force.device
            ),
            "deltas": deltas,
        }

    @staticmethod
    def _force_pre_kwargs(state: Optional[dict]) -> dict:
        if state is None:
            return {}
        return {
            "force_history": state["history"],
            "force_history_mask": state["history_mask"],
            "noisy_force_future": state["latents"],
            "force_future_mask": state["future_mask"],
        }

    @staticmethod
    def _force_infer_kwargs(force_history, force_history_mask) -> dict:
        """Keep a model without force conditioning out of the force code path.

        An empty dict means the callee uses its own defaults; half a force input
        is a bug either way, so a one-sided pair is passed through and rejected
        by `_build_force_infer_state`.
        """
        if force_history is None and force_history_mask is None:
            return {}
        return {
            "force_history": force_history,
            "force_history_mask": force_history_mask,
        }

    def _advance_force_state(self, state: dict, pred_force: torch.Tensor, step_index: int) -> dict:
        state["latents"] = self.infer_action_scheduler.step(
            pred_force, state["deltas"][step_index], state["latents"]
        )
        return state

    def _decode_force_state(self, state: Optional[dict]) -> Optional[dict]:
        """Return the sampled force trajectory in physical units for logging."""
        if state is None:
            return None
        raw = self.force_conditioner["history_encoder"].denormalize(state["latents"])
        return {
            "force_future": state["latents"][0].detach().to(device="cpu", dtype=torch.float32),
            "force_future_newton": raw[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @staticmethod
    def _apply_force_feedback(action: torch.Tensor, force: Optional[dict], feedback) -> dict:
        """P4: hand the sampled force trajectory to an optional external controller.

        Disabled by default -- `feedback is None` returns the policy action
        untouched, byte for byte. There is no learned corrector here because the
        corpus has no (policy action, corrected action) pairs, and inventing a
        force-error-to-torque map would be exactly the unvalidated controller the
        task forbids. Whatever is plugged in must be a pure function of the
        predicted force and the proposed chunk, and its output is checked before
        it can replace the action.

        Args:
            action: `[T, action_dim]` float32 CPU tensor, the sampled chunk.
            force: the `_decode_force_state` payload; `force_future_newton` is in
                newtons, `force_future` in model space.
            feedback: callable `(action, force) -> [T, action_dim] tensor`.

        Returns:
            dict with `action` (possibly corrected), `action_raw` and
            `force_feedback_applied`.
        """
        out = {"action": action, "action_raw": action, "force_feedback_applied": False}
        if feedback is None:
            return out
        if force is None:
            raise ValueError(
                "`force_feedback` was supplied but there is no predicted force trajectory: this "
                "model has no force conditioning, or `force_history` was omitted at infer time."
            )
        if not callable(feedback):
            raise TypeError(f"`force_feedback` must be callable, got {type(feedback)}")
        corrected = feedback(action, force)
        if not isinstance(corrected, torch.Tensor):
            raise TypeError(
                f"`force_feedback` must return a tensor, got {type(corrected)}"
            )
        if corrected.shape != action.shape:
            raise ValueError(
                f"`force_feedback` returned {tuple(corrected.shape)} for an action chunk of "
                f"{tuple(action.shape)}; a correction may not change the chunk's shape."
            )
        corrected = corrected.detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(corrected).all().item()):
            raise ValueError(
                "`force_feedback` returned NaN or Inf; refusing to execute an action chunk "
                "that is not finite."
            )
        out["action"] = corrected
        out["action_delta_max"] = float((corrected - action).abs().max().item())
        out["force_feedback_applied"] = True
        return out

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
        force_state: Optional[dict] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            **self._force_pre_kwargs(force_state),
        )

        attention_masks = self._build_mot_attention_masks(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_masks,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        pred_force = None
        if force_state is not None:
            force_tokens = self.action_expert.apply_force_injection_post_dit(
                tokens_out["action"], action_pre
            )
            pred_force = self.action_expert.predict_force_velocity(force_tokens, action_pre)
        return pred_video, pred_action, pred_force

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_masks = self._build_mot_attention_masks(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_masks,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: Union[torch.Tensor, Sequence[torch.Tensor]],
        video_seq_len: int,
        force_state: Optional[dict] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
            **self._force_pre_kwargs(force_state),
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        pred_force = None
        if force_state is not None:
            force_tokens = self.action_expert.apply_force_injection_post_dit(action_tokens, action_pre)
            pred_force = self.action_expert.predict_force_velocity(force_tokens, action_pre)
        return self.action_expert.post_dit(action_tokens, action_pre), pred_force

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
        force_history: Optional[torch.Tensor] = None,
        force_history_mask: Optional[torch.Tensor] = None,
        force_feedback=None,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
                **FastWAM._force_infer_kwargs(force_history, force_history_mask),
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        force_state = self._build_force_infer_state(
            force_history=force_history,
            force_history_mask=force_history_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        for step_idx, (step_t_video, step_delta_video, step_t_action, step_delta_action) in enumerate(zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        )):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi, pred_force_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
                force_state=force_state,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()
            if force_state is not None:
                force_state = self._advance_force_state(force_state, pred_force_posi, step_idx)

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        out = {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }
        if force_state is not None:
            out.update(self._decode_force_state(force_state))
        out.update(self._apply_force_feedback(action_out, out, force_feedback))
        return out

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_masks = self._build_mot_attention_masks(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_attention_mask = self._build_video_attention_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=video_attention_mask,
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action, _pred_force = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_masks,
                video_seq_len=video_seq_len,
            )

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        force_history: Optional[torch.Tensor] = None,
        force_history_mask: Optional[torch.Tensor] = None,
        force_feedback=None,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            force_history=force_history,
            force_history_mask=force_history_mask,
            force_feedback=force_feedback,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.force_conditioner is not None:
            payload["force_calibration"] = force_calibration_fingerprint(self)
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def _force_calibration_audit(self, declared, path):
        """Refuse force weights whose declared units differ from this model's.

        `mot` carries `force_mean`/`force_std` as buffers, so a published checkpoint
        brings its own calibration with it. That is only safe if it is the
        calibration the weights were trained under: if
        `force.norm_stats.file` was regenerated after publication, loading would
        silently re-unit every force the trained modules expect (§四.6). The
        publisher therefore records a fingerprint, and a checkpoint that carries
        force weights without one is refused rather than trusted.
        """
        encoder = self.force_conditioner["history_encoder"]
        live = force_calibration_fingerprint(self)
        if declared is None:
            raise RuntimeError(
                "%s restores force weights but declares no `force_calibration` "
                "fingerprint, so its force units cannot be verified against this "
                "model's calibration (%s)." % (path, live)
            )
        if declared != live:
            diff = {k: (declared.get(k), live.get(k)) for k in set(declared) | set(live)
                    if declared.get(k) != live.get(k)}
            raise RuntimeError(
                "%s was published under force units %s, but this model's calibration is "
                "%s; refusing to run inference with mismatched force normalization."
                % (path, diff, live)
            )
        logger.info("force calibration verified for %s: %s", path, live)
        return live

    def _report_force_load_status(self, missing, unexpected, path, declared=None):
        """State which force tensors a `strict=False` load did or did not restore.

        Loading the upstream base into a force-conditioned model is supposed to
        leave the new tensors at their initialization, but `strict=False` hides
        that, and r4/r5 lost `proprio_encoder` exactly this way. So the split is
        reported explicitly, and any force key the *model* does not have is an
        error rather than a warning.
        """
        conditioner = self.force_conditioner
        if conditioner is None:
            stray = [key for key in unexpected if "force_conditioner" in key]
            if stray:
                raise RuntimeError(
                    f"{path} contains force-conditioning weights but this model was built "
                    f"without `force_config`: {stray[:5]}"
                )
            return
        prefix = "mixtures.action.force_conditioner."
        model_force_keys = {
            f"{prefix}{name}"
            for name, _ in conditioner.named_parameters()
        } | {
            f"{prefix}{name}"
            for name, _ in conditioner.named_buffers()
        }
        # `missing` is every model key the payload lacked, LoRA adapters included,
        # so only the force-prefixed part of it may be subtracted here.
        missing_force = [key for key in missing if key.startswith(prefix)]
        restored = len(model_force_keys) - len(missing_force)
        if restored == 0:
            logger.warning(
                "%s has no force-conditioning weights; all %d force tensors stay at "
                "initialization (expected only when loading a pre-force base).",
                path, len(model_force_keys),
            )
        else:
            logger.info(
                "force conditioning restored from %s: %d/%d tensors",
                path, restored, len(model_force_keys),
            )
        encoder = self.force_conditioner["history_encoder"]
        if restored and float(encoder.stats_are_set.detach().float().item()) != 1.0:
            raise RuntimeError(
                f"{path} restored force weights but not the force calibration buffers; "
                "refusing to train or infer with identity normalization."
            )
        if restored:
            self._force_calibration_audit(declared, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "mot" in payload:
            missing, unexpected = self.mot.load_state_dict(payload["mot"], strict=False)
            self._report_force_load_status(missing, unexpected, path,
                                           declared=payload.get("force_calibration"))
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
