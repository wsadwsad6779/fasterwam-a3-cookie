import hydra
from omegaconf import DictConfig
from fasterwam.runtime import build_datasets
from fasterwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="/workspace/FasterWAM/configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    train_ds, val_ds = build_datasets(cfg.data)
    print("train_len", len(train_ds), "val_len", None if val_ds is None else len(val_ds))
    s = train_ds[0]
    for k in sorted(s.keys()):
        v = s[k]
        print(k, type(v).__name__, getattr(v, "shape", None), getattr(v, "dtype", None))
    print("DS_CHECK_OK")


main()
