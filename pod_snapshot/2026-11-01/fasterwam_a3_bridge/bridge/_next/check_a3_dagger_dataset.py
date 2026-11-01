"""Instantiate the real training dataset on the DAgger corrective set.

Checks the things that would silently break a LoRA run: exact frames.dat size,
episode window count, sample keys/shapes, and how far the new labels sit
outside the base dataset's normalisation ranges.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

DS = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/a3_data/a3_fw_dagger")
BASE_STATS = Path("/workspace/a3_data/a3_fw/dataset_stats.json")
CFG = Path("/workspace/fasterwam-a3-cookie/finetune/a3_lora_r1/config.yaml")


def stats_table(name, key):
    st = json.loads(BASE_STATS.read_text())[name][key]
    return {k: np.asarray(v, dtype=np.float64) for k, v in st.items() if isinstance(v, list)}


def out_of_range(prefix, values, st):
    lo, hi = st["global_min"], st["global_max"]
    q01, q99 = st["global_q01"], st["global_q99"]
    return {
        f"{prefix}_beyond_minmax_pct": round(100.0 * float(((values < lo) | (values > hi)).mean()), 2),
        f"{prefix}_beyond_q0101_pct": round(100.0 * float(((values < q01) | (values > q99)).mean()), 2),
    }


def main():
    arr = np.load(DS / "arrays.npz")
    n = arr["action"].shape[0]
    bytes_needed = n * 224 * 448 * 3
    bytes_have = (DS / "frames.dat").stat().st_size
    print(f"[check] transitions={n} frames_bytes={bytes_have} expected={bytes_needed} ok={bytes_have == bytes_needed}")
    print(f"[check] episodes={arr['ep_len'].tolist()} starts={arr['ep_start'].tolist()}")

    node = OmegaConf.load(CFG).data.train
    node.dataset_dirs = [str(DS)]
    node.pretrained_norm_stats = str(DS / "dataset_stats.json")
    node.is_training_set = False
    ds = instantiate(node)
    print(f"[check] dataset windows={len(ds._windows)} frames={tuple(ds._frames.shape)}")

    sample = ds[0]

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{prefix}.{k}" if prefix else str(k))
        elif isinstance(obj, torch.Tensor):
            flat = obj.float()
            print(f"[sample] {prefix:<28} {tuple(obj.shape)!s:<18} {str(obj.dtype):<10} "
                  f"min={flat.min():.3f} max={flat.max():.3f} mean={flat.mean():.3f}")
        else:
            print(f"[sample] {prefix:<28} {type(obj).__name__} {str(obj)[:60]}")

    walk(sample)

    st = stats_table("action", "default")
    act = arr["action"].astype(np.float64)
    prop = arr["proprio"].astype(np.float64)
    for i in range(7):
        print(f"[range] action[{i}] base=[{st['global_min'][i]:.4f},{st['global_max'][i]:.4f}] "
              f"dagger=[{act[:, i].min():.4f},{act[:, i].max():.4f}]")
    print("[range]", out_of_range("action", act, st))
    root = json.loads(BASE_STATS.read_text())
    if "state" in root:
        pst = stats_table("state", "default")
        print("[range]", out_of_range("state", prop, pst))

    g = arr["action"][:, 6]
    segs = [int(np.sum(np.sign(g[a:a + l])[1:] != np.sign(g[a:a + l])[:-1]))
            for a, l in zip(arr["ep_start"], arr["ep_len"])]
    print(f"[grip] plus_one_frac={float((g > 0).mean()):.3f} close_open_switches_per_ep={segs}")


if __name__ == "__main__":
    main()
