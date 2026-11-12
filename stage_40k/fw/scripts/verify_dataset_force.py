"""Prove the training batch carries the force windows, before any model is built.

Stage 1 of the rehearsal died with `sample['force_history'] is required` while the
dataset had loaded the force table fine: `RobotVideoDataset._get` rebuilds every
sample as a fixed-key dict, so the per-window force tensors were dropped between
the dataset and the trainer. This script checks that seam on the real corpus with
no GPU and no model, and includes the negative controls.
"""
import atexit
import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import default_collate

from fasterwam.runtime import build_datasets  # noqa: F401  (parity with the trainer)
from fasterwam.datasets.lerobot.a3_video_dataset import FORCE_SAMPLE_KEYS

def _checks():
    rec = {"checks": [], "events": []}

    def check(name, ok, detail=None):
        rec["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
        if not ok:
            raise AssertionError("%s: %s" % (name, detail))

    return rec, check


def _writer(out_path):
    """Returns finish(rec); the report is written once, even if a check raises."""
    state = {"done": False}

    def finish(rec):
        if state["done"]:
            return rec
        state["done"] = True
        payload = dict(rec)
        payload["passed"] = sum(1 for c in rec["checks"] if c["ok"])
        payload["total"] = len(rec["checks"])
        payload["failed"] = [c["name"] for c in rec["checks"] if not c["ok"]]
        text = json.dumps(payload, indent=2, sort_keys=True, default=str)
        if out_path is not None:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_name(out_path.name + ".tmp")
            tmp.write_text(text)
            tmp.replace(out_path)
        print(text[-4000:])
        print("dataset force wiring: %d/%d checks passed; failed=%s"
              % (payload["passed"], payload["total"], payload["failed"]))
        return payload

    return finish


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    out_path = cfg.get("report")
    out_path = Path(str(out_path)) if out_path not in (None, "None") else None
    rec, check = _checks()
    finish = _writer(out_path)
    atexit.register(lambda: finish(rec))

    ds, _ = build_datasets(cfg.data)
    spec = {"hist_len": int(cfg.data.train.force_hist_len),
            "horizon": int(cfg.data.train.force_horizon),
            "channels": [int(c) for c in cfg.data.train.force_channels],
            "force_dim": len(cfg.data.train.force_channels)}
    rec["spec"] = spec
    rec["dataset_len"] = len(ds)
    rec["force_loaded"] = ds._force is not None
    check("D0_force_table_loaded", ds._force is not None, "ds._force is None")
    check("D1_config_and_dataset_agree",
              (ds._force_hist_len, ds._force_horizon, ds._force_channels)
              == (spec["hist_len"], spec["horizon"], spec["channels"]),
              "dataset %s/%s/%s vs config %s/%s/%s"
              % (ds._force_hist_len, ds._force_horizon, ds._force_channels,
                 spec["hist_len"], spec["horizon"], spec["channels"]))

    # The model's own view of the frozen geometry, read off the config it will be
    # built from, so a dataset/model disagreement cannot hide here.
    m_spec = cfg.model.force.conditioner
    check("D2_model_spec_matches_dataset",
              [int(c) for c in m_spec.channels] == spec["channels"]
              and int(m_spec.history_encoder.hist_len) == spec["hist_len"]
              and int(m_spec.future_encoder.horizon) == spec["horizon"]
              and int(m_spec.history_encoder.force_dim) == spec["force_dim"]
              and int(m_spec.future_encoder.force_dim) == spec["force_dim"],
              "model %s/%d/%d/fdim %s vs dataset %s/%d/%d/%d"
              % ([int(c) for c in m_spec.channels], int(m_spec.history_encoder.hist_len),
                 int(m_spec.future_encoder.horizon), int(m_spec.history_encoder.force_dim),
                 spec["channels"],
                 spec["hist_len"], spec["horizon"], spec["force_dim"]))

    probe_idx = sorted({0, 1, len(ds) - 1, len(ds) // 2, int(0.9 * len(ds))})
    samples = []
    for idx in probe_idx:
        sample = ds[idx]
        samples.append((idx, sample))
        missing = [k for k in FORCE_SAMPLE_KEYS if k not in sample]
        check("D3_keys_present@%d" % idx, not missing, "missing %s" % missing)

        hist = sample["force_history"]
        hm = sample["force_history_mask"]
        fut = sample["force_future"]
        fm = sample["force_future_mask"]
        ndim, nch = spec["hist_len"], spec["force_dim"]
        check("D4_shapes@%d" % idx,
                  tuple(hist.shape) == (ndim, nch) and tuple(fut.shape) == (spec["horizon"], nch)
                  and tuple(hm.shape) == (ndim,) and tuple(fm.shape) == (spec["horizon"],),
                  "hist %s hm %s fut %s fm %s" % (tuple(hist.shape), tuple(hm.shape),
                                                  tuple(fut.shape), tuple(fm.shape)))
        check("D5_bool_masks@%d" % idx,
                  hm.dtype == torch.bool and fm.dtype == torch.bool,
                  "%s / %s" % (hm.dtype, fm.dtype))
        check("D6_padded_rows_zero@%d" % idx,
                  float(hist[~hm].abs().max().item() if bool((~hm).any()) else 0.0) == 0.0
                  and float(fut[~fm].abs().max().item() if bool((~fm).any()) else 0.0) == 0.0,
                  "a masked row carries a nonzero force value")

        # History is a left-padded *trailing suffix*: the True run must end at the
        # last row. Future is a right-padded *leading prefix*: starts at row 0.
        h_idx = hm.nonzero(as_tuple=True)[0]
        f_idx = fm.nonzero(as_tuple=True)[0]
        check("D7_history_is_trailing_suffix@%d" % idx,
                  h_idx.numel() > 0 and int(h_idx[-1]) == ndim - 1
                  and bool(torch.equal(h_idx, torch.arange(ndim - h_idx.numel(), ndim))),
                  "history True rows %s" % (h_idx.tolist()[:20],))
        check("D8_future_is_leading_prefix@%d" % idx,
                  f_idx.numel() > 0 and int(f_idx[0]) == 0
                  and bool(torch.equal(f_idx, torch.arange(f_idx.numel()))),
                  "future True rows %s" % (f_idx.tolist()[:20],))

    # Independent read path: the dataset's own `force_rows()` against the source
    # table. A window that borrowed from a neighbouring episode, or was shifted by
    # one frame, fails here rather than being compared against itself.
    for idx, sample in samples:
        s = int(sample["window_offset"])
        ep_start = int(ds._window_ep_start[idx])
        ep_end = int(ds._window_ep_end[idx])
        hist_len = int(sample["force_history_mask"].sum())
        want_hist = ds.force_rows(list(range(s - hist_len + 1, s + 1)))
        got_hist = sample["force_history"][spec["hist_len"] - hist_len:].numpy()
        check("D9_history_equals_source@%d" % idx,
                  bool((want_hist == got_hist).all()),
                  "max abs diff %.6g" % float(abs(want_hist - got_hist).max()))
        fut_len = int(sample["force_future_mask"].sum())
        want_fut = ds.force_rows(list(range(s + 1, s + 1 + fut_len)))
        got_fut = sample["force_future"][:fut_len].numpy()
        check("D10_future_equals_source@%d" % idx,
                  bool((want_fut == got_fut).all()),
                  "max abs diff %.6g" % float(abs(want_fut - got_fut).max()))
        # No leakage: the earliest history row is the first row of the window's own
        # episode at worst, and the last history row is frame s, never s+1.
        check("D11_history_within_episode@%d" % idx, s - hist_len + 1 >= ep_start,
                  "history reached %d, episode starts %d" % (s - hist_len + 1, ep_start))
        check("D12_future_within_episode@%d" % idx, s + fut_len < ep_end,
                  "future reached %d, episode ends (exclusive) %d" % (s + fut_len, ep_end))
        check("D13_future_starts_after_current@%d" % idx,
                  s + 1 <= s + fut_len and fut_len == min(spec["horizon"], ep_end - 1 - s),
                  "fut_len %d expected %d" % (fut_len, min(spec["horizon"], ep_end - 1 - s)))

    batch = default_collate([sample for _, sample in samples])
    rec["batch_shapes"] = {k: list(batch[k].shape) for k in FORCE_SAMPLE_KEYS
                           if torch.is_tensor(batch[k])}
    check("D14_collate_batch_dims",
              batch["force_history"].shape == (len(samples), spec["hist_len"], spec["force_dim"])
              and batch["force_future"].shape == (len(samples), spec["horizon"], spec["force_dim"])
              and batch["force_history_mask"].shape == (len(samples), spec["hist_len"])
              and batch["force_history_mask"].dtype == torch.bool,
              rec["batch_shapes"])

    # Negative controls: the hook must fail loudly rather than hand back a sample
    # with no force, and the model must refuse a batch that lost a key.
    raised = None
    try:
        ds._extra_sample_fields({"unrelated": torch.zeros(1)})
    except KeyError as exc:
        raised = str(exc)
    check("N1_hook_raises_when_keys_missing", raised is not None, raised)
    rec["n1_message"] = raised

    stripped = dict(samples[0][1])
    stripped.pop("force_future_mask")
    raised = None
    try:
        _require_like_model(stripped, spec)
    except ValueError as exc:
        raised = str(exc)
    check("N2_model_refuses_missing_key", raised is not None, raised)
    rec["n2_message"] = raised

    short = dict(samples[0][1])
    short["force_history"] = short["force_history"][:spec["hist_len"] - 1]
    short["force_history_mask"] = short["force_history_mask"][:spec["hist_len"] - 1]
    raised = None
    try:
        _require_like_model(short, spec)
    except ValueError as exc:
        raised = str(exc)
    check("N3_model_refuses_wrong_hist_len", raised is not None, raised)
    rec["n3_message"] = raised

    payload = finish(rec)
    if payload["failed"]:
        sys.exit(1)


def _require_like_model(sample, spec):
    """Mirror of FastWAM._require_force_training_inputs' shape/key gate.

    Re-implemented rather than called so this script needs no model and no GPU;
    the rehearsal then exercises the real method on the real batch.
    """
    for key in ("force_history", "force_history_mask", "force_future", "force_future_mask"):
        if key not in sample:
            raise ValueError("sample['%s'] is required" % key)
    hist, fut = sample["force_history"], sample["force_future"]
    if hist.ndim == 3:
        hist = hist[0]
        fut = fut[0]
    if hist.shape[0] != spec["hist_len"]:
        raise ValueError("force_history must carry %d steps, got %d"
                         % (spec["hist_len"], hist.shape[0]))
    if fut.shape[0] != spec["horizon"]:
        raise ValueError("force_future must carry %d steps, got %d"
                         % (spec["horizon"], fut.shape[0]))
    if hist.shape[1] != spec["force_dim"]:
        raise ValueError("wrong force width")


if __name__ == "__main__":
    main()
