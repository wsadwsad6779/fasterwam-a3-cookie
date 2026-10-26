"""Merge LoRA adapters from a training checkpoint back into the base weights.

Writes a checkpoint with the same layout as the released FasterWAM checkpoints
(`mot` / `proprio_encoder` / `step`), so the existing inference worker loads it
without any LoRA support.
"""
import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from fasterwam.lora import apply_lora, merge_lora
from fasterwam.runtime import _mixed_precision_to_model_dtype, _normalize_mixed_precision
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision))
    model = instantiate(cfg.model, model_dtype=dtype, device=device)

    n_layers, n_trainable = apply_lora(model, cfg.lora)
    print(f"applied lora to {n_layers} layers ({n_trainable} params)")

    payload = torch.load(cfg.lora_ckpt, map_location=device)
    missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
    print(f"loaded mot: missing={len(missing)} unexpected={len(unexpected)}")
    if model.proprio_encoder is not None and "proprio_encoder" in payload:
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)

    merged = merge_lora(model)
    out = {
        "mot": model.mot.state_dict(),
        "step": payload.get("step"),
        "torch_dtype": str(dtype),
    }
    if model.proprio_encoder is not None:
        out["proprio_encoder"] = model.proprio_encoder.state_dict()
    torch.save(out, cfg.output)
    print(f"merged {merged} lora layers -> {cfg.output}")


if __name__ == "__main__":
    main()
