"""CPU-only, no-model probe: is the live proprio inside the band the stats file recorded?

The 40k smoke's episode-11 record says `proprio_base_min_all == proprio_base_max_all == -5.0`:
in all sixty replans, every one of the eight base proprio columns came out of the deployment
normaliser at exactly its clamp floor, which is the signature of a value at least two band-widths
below the recorded minimum. If that is what the model was fed, the rollout measured nothing about
the checkpoint, and the 160-episode batch would be 160 episodes of the same mistake.

Nothing here opens a checkpoint or touches the GPU. The claim is arithmetic, and both sides of it
are already on disk:

  * the band -- `a3_fw_v1g/dataset_stats.json`, the file `run40k/config.yaml` names as
    `pretrained_norm_stats`, read through the loader the deployment worker uses;
  * the distribution inside the band -- the 42,681 raw 14-wide states in `a3_fw_v1g/arrays.npz`,
    i.e. exactly what training normalised;
  * the live side -- one real A3 observation from `env_factory.reset(seed=11)`, put through the
    worker's own `build_proprio` and the harness's `build_proprio14`, then through the worker's own
    `normalize_proprio`.

So the output answers one question with numbers: for each of the fourteen columns, what is the
corpus range, what is the band, what does the live observation say, and does the live value map
inside [-1, 1] the way the corpus values do. `normalize_proprio` is also run over corpus rows, so a
normaliser that was handed the wrong band shows up as corpus rows failing to map, not as a guess.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import eval40k as E  # noqa: E402


def per_col(a):
    """min / p01 / p50 / p99 / max of a (rows, cols) array, as plain lists."""
    a = np.asarray(a, dtype=np.float64)
    return {"shape": list(a.shape),
            "min": [float(x) for x in a.min(0)],
            "p01": [float(x) for x in np.percentile(a, 1, axis=0)],
            "p50": [float(x) for x in np.percentile(a, 50, axis=0)],
            "p99": [float(x) for x in np.percentile(a, 99, axis=0)],
            "max": [float(x) for x in a.max(0)]}


def band_mapped(x, lo, hi):
    """What min/max mode is supposed to compute, before the clamp: [-1, 1] for x in [lo, hi]."""
    x = np.asarray(x, dtype=np.float64)
    return (2.0 * (x - lo) / (hi - lo)) - 1.0


def main():
    out = {}
    calib = json.loads(Path(E.CALIB_PATH).read_text())
    zero = np.asarray(calib["zero_load_ref_median"], dtype=np.float64)
    std = np.asarray(calib["std"], dtype=np.float64)
    out["calib"] = {"path": str(E.CALIB_PATH), "version": calib.get("version"),
                    "zero_load_ref_median": zero.tolist(), "std": std.tolist(),
                    "mean": [float(x) for x in calib["mean"]]}

    raw_stats = json.loads(Path(E.V1G_STATS).read_text())
    entry = raw_stats["state"]["default"]
    lo = np.asarray(entry["global_min"], dtype=np.float64)
    hi = np.asarray(entry["global_max"], dtype=np.float64)
    out["stats"] = {"path": str(E.V1G_STATS), "state_keys": sorted(raw_stats["state"]),
                    "band_min": lo.tolist(), "band_max": hi.tolist(),
                    "band_width": (hi - lo).tolist(),
                    "global_mean": ([float(x) for x in entry["global_mean"]]
                                    if "global_mean" in entry else None),
                    "global_std": ([float(x) for x in entry["global_std"]]
                                   if "global_std" in entry else None)}

    # ---- the corpus side: what training actually normalised -------------------------------
    v1g = np.load(str(E.V1G_ARRAYS))
    p14 = np.asarray(v1g["proprio"], dtype=np.float64)
    out["corpus_raw_14"] = per_col(p14)
    rng = np.random.default_rng(20261112)
    sample = p14[rng.integers(0, p14.shape[0], 512)]
    mapped = band_mapped(sample, lo, hi)
    out["corpus_band_check"] = {
        "rows_sampled": int(sample.shape[0]),
        "base_columns_inside_band_frac": float(np.mean(np.all(np.abs(mapped[:, :E.PROPRIO_BASE_DIM])
                                                              <= 1.0 + 1e-6, axis=1))),
        "per_col_min_mapped": [float(x) for x in mapped.min(0)],
        "per_col_max_mapped": [float(x) for x in mapped.max(0)],
        "whole_corpus_base_outside_band_rows": None,
    }
    all_mapped = np.clip(band_mapped(p14[:, :E.PROPRIO_BASE_DIM], lo[:E.PROPRIO_BASE_DIM],
                                     hi[:E.PROPRIO_BASE_DIM]), -5, 5)
    out["corpus_band_check"]["whole_corpus_base_outside_band_rows"] = int(
        np.sum(np.any(np.abs(all_mapped) > 1.0 + 1e-3, axis=1)))
    out["corpus_band_check"]["whole_corpus_rows"] = int(p14.shape[0])

    # ---- the live side: one real observation, the worker's own functions ------------------
    env_factory, W = E.build_env(None)
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

    doc = OmegaConf.load(str(E.DEFAULT_CFG))
    stats = load_dataset_stats_from_json(str(E.V1G_STATS))
    E.STATS_FACTS.clear()
    E.STATS_FACTS.update(E.check_stats_width(stats, E.V1G_STATS, calib_std=std))
    processor = instantiate(doc.data.train.processor).eval()
    processor.set_normalizer_from_stats(stats)
    out["processor"] = {"class": type(processor).__name__,
                        "norm_default_mode": str(processor.norm_default_mode),
                        "proprio_output_dim": int(processor.proprio_output_dim),
                        "num_obs_steps": int(processor.num_obs_steps),
                        "state_raw_shape": [int(m["raw_shape"])
                                            for m in processor.shape_meta["state"]]}

    env = env_factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=False)
    obs, _info = env_factory.reset(env, 11, randomize_cookies=True,
                                   randomize_boxes=False, batch_offset_m=0.0)
    obsd = obs if isinstance(obs, dict) else obs[0]
    keys = sorted(obsd)
    out["obs"] = {"keys": keys,
                 "eef_pose": [float(x) for x in
                              np.asarray(obsd["observation.eef_pose"], dtype=np.float64).reshape(-1)],
                 "state": [float(x) for x in
                           np.asarray(obsd["observation.state"], dtype=np.float64).reshape(-1)][:14],
                 "force18": [float(x) for x in
                             np.asarray(obsd["observation.force"], dtype=np.float64).reshape(-1)]}

    base8 = np.asarray(W.build_proprio(obsd), dtype=np.float64).reshape(-1)
    p14_live = np.asarray(E.build_proprio14(obsd, W, calib), dtype=np.float64).reshape(-1)
    norm_live = np.asarray(W.normalize_proprio(p14_live.astype(np.float32), processor),
                           dtype=np.float64).reshape(-1)
    norm_corpus = np.asarray(W.normalize_proprio(p14[0].astype(np.float32), processor),
                             dtype=np.float64).reshape(-1)
    out["live"] = {"base8_worker_build_proprio": base8.tolist(),
                   "proprio14_raw": p14_live.tolist(),
                   "proprio14_facts": {k: v for k, v in sorted(E.PROPRIO14_FACTS.items())},
                   "normalised_live": norm_live.tolist(),
                   "normalised_corpus_row0": norm_corpus.tolist(),
                   "normalised_live_at_clamp": [bool(abs(x) >= 4.999) for x in norm_live]}
    out["live_vs_band"] = {
        "mapped_no_clamp": [float(x) for x in band_mapped(p14_live, lo, hi)],
        "band_min": lo.tolist(), "band_max": hi.tolist(),
        "how_far_below_min_in_band_widths":
            [float((lo[i] - p14_live[i]) / (hi[i] - lo[i])) for i in range(14)],
        "corpus_p50": out["corpus_raw_14"]["p50"],
    }
    # the same observation as the model-side fold computed it, so the aggregate names in the
    # smoke record can be re-derived here instead of inferred
    mn, mx = float(norm_live.min()), float(norm_live.max())
    out["fold_reproduction"] = {
        "proprio_min": mn, "proprio_max": mx,
        "proprio_base_min": float(norm_live[:E.PROPRIO_BASE_DIM].min()),
        "proprio_base_max": float(norm_live[:E.PROPRIO_BASE_DIM].max()),
        "proprio_force_min": float(norm_live[E.PROPRIO_BASE_DIM:].min()),
        "proprio_force_max": float(norm_live[E.PROPRIO_BASE_DIM:].max()),
        "width": int(norm_live.shape[0]),
        "identity_max_equals_max_of_groups": mx == max(
            float(norm_live[:E.PROPRIO_BASE_DIM].max()),
            float(norm_live[E.PROPRIO_BASE_DIM:].max())),
    }
    if hasattr(env, "close"):
        env.close()
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
