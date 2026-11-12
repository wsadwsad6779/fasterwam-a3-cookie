"""`scripts/train_light.py` plus the r5 learning-rate restart.

    accelerate launch --num_processes 1 scripts/train_light_r5.py \
        data=a3_2cam task=a3_fasterwam batch_size=6 max_steps=8000 \
        learning_rate=5e-5 log_every=20 save_every=500 eval_every=0 \
        output_dir=/workspace/a3_train_runs/a3_lora_r5 \
        resume=/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000

`r4_patch.install()` first (sampler pass handling + light-state save/resume),
then `r5_patch.install()`, which only acts when `FW_LR_RESTART=1`. No upstream
file is edited by either.
"""
import hydra
from omegaconf import DictConfig

import fasterwam.r4_patch as r4_patch
import fasterwam.r5_patch as r5_patch
from fasterwam.runtime import run_training
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
r4_patch.install()
r5_patch.install()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
