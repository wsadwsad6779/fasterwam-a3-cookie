"""`scripts/train.py` + the r4 runtime patch, without editing upstream files.

    accelerate launch ... scripts/train_light.py data.train.dataset_dirs=... \
        resume=<base .pt | light-state dir> ...
"""
import hydra
from omegaconf import DictConfig

import fasterwam.r4_patch as r4_patch
from fasterwam.runtime import run_training
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()
r4_patch.install()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
