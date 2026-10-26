"""Dependency-free LoRA for the Wan-2.2 DiT attention projections.

`LoRALinear` subclasses `nn.Linear` and reuses the original weight/bias
Parameters, so state_dict keys stay `<block>.self_attn.q.weight` (base) plus
`<block>.self_attn.q.lora_A/B` (adapters). That lets a plain model load a LoRA
checkpoint with strict=False and lets `merge_lora` produce a checkpoint that is
byte-for-byte compatible with the inference path.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Linear):
    def __init__(self, base: nn.Linear, r: int, alpha: int, dropout: float = 0.0):
        super().__init__(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )
        self.weight = base.weight
        if base.bias is not None:
            self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r = int(r)
        self.scaling = float(alpha) / float(r)
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(self.r, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(base.out_features, self.r, device=base.weight.device, dtype=base.weight.dtype)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        delta = F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B)
        return out + delta * self.scaling


def apply_lora(model, cfg) -> tuple[int, int]:
    """Wrap matching `nn.Linear` leaves of the named experts. Returns (layers, trainable)."""
    experts = [str(e) for e in cfg.get("experts", ["video_expert", "action_expert"])]
    leaves = {str(t) for t in cfg.get("target_modules", ["q", "k", "v", "o"])}
    r = int(cfg.get("r", 16))
    alpha = int(cfg.get("alpha", 32))
    dropout = float(cfg.get("dropout", 0.0))

    n_replaced = 0
    for expert_name in experts:
        expert = getattr(model, expert_name, None)
        if expert is None:
            raise ValueError(f"LoRA expert not found on model: {expert_name}")
        for name, child in list(expert.named_modules()):
            if not isinstance(child, nn.Linear) or isinstance(child, LoRALinear):
                continue
            if name.rsplit(".", 1)[-1] not in leaves:
                continue
            parent_path, _, leaf = name.rpartition(".")
            parent = expert.get_submodule(parent_path) if parent_path else expert
            setattr(parent, leaf, LoRALinear(child, r, alpha, dropout))
            n_replaced += 1

    n_trainable = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    return n_replaced, n_trainable


def merge_lora(module: nn.Module) -> int:
    """Fold every LoRA delta into its base weight and restore plain `nn.Linear`."""
    merged = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            with torch.no_grad():
                child.weight.add_((child.lora_B @ child.lora_A) * child.scaling)
            linear = nn.Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
                dtype=child.weight.dtype,
            )
            linear.weight = child.weight
            if child.bias is not None:
                linear.bias = child.bias
            setattr(module, name, linear)
            merged += 1
        else:
            merged += merge_lora(child)
    return merged
