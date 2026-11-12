"""Segmented 40k training entry: r4 sampler fixes + v3 light state + FW_STOP_STEP.

Force-conditioned run (V1):

    accelerate launch --num_processes 1 scripts/train_seg.py \
        data=a3_2cam_force task=a3_force_40k batch_size=6 max_steps=40000 \
        learning_rate=1e-4 log_every=20 save_every=1000 eval_every=0 \
        output_dir=<run> resume=<base .pt | previous light-state dir>

``max_steps`` is always the full 40000 so the cosine is built once; the env var
``FW_STOP_STEP`` says where this segment ends. Never set ``FW_LR_RESTART``.

Patches are installed before Hydra touches the modules they wrap, so the order
here is load-bearing: the r4 sampler fixes first, then the frozen-base load
guard, then the segmented stop step.
"""
import hydra
from omegaconf import DictConfig

import fasterwam.base_load_guard as base_load_guard
import fasterwam.r4_patch as r4_patch
import fasterwam.seg_patch as seg_patch
from fasterwam.runtime import run_training
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
r4_patch.install()
base_load_guard.install()
seg_patch.install()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
