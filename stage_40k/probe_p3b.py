"""Why do the live force buffers differ from the calibration JSON by 2.5e-02?

Stage 5 of the rehearsal compared `force_mean`/`force_std` as they sit in the
published checkpoint against `force_calib_v1.json` with `atol=1e-5` and failed:
max abs error 0.0255 on the mean and 0.0019 on the std. Two explanations have
completely different consequences, and only one of them is a defect:

* the model holds the buffers in the *model dtype* (bf16), so the effective
  constants are the JSON rounded to bf16. Training and inference both round the
  same way, the fingerprint covers the rounded values, and nothing is wrong --
  the check just asked for float32 agreement out of a bf16 buffer;
* the training config points at a *different* JSON than the one stage 5 passed,
  in which case the published weights are calibrated in units nobody froze.

This separates them without loading the 11 GB file: the fingerprint is a sha256
of the buffer widened to float32, so re-deriving it from each candidate JSON
under each candidate rounding identifies both the file and the dtype that the
real run actually used.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import torch

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def digest(values, dtype):
    t = torch.tensor(values, dtype=torch.float32).to(dtype).to(torch.float32).contiguous()
    return hashlib.sha256(t.numpy().tobytes()).hexdigest(), t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", required=True, help="the fingerprint the real run recorded")
    ap.add_argument("--glob", default="/workspace/a3_data/a3_fw_force/*.json")
    ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()

    want = json.loads(Path(a.want).read_text())["published_force_calibration"]
    print("want ::", json.dumps(want, sort_keys=True))
    cands = sorted(set(glob.glob(a.glob)) | set(a.extra))
    rows = []
    for path in cands:
        doc = json.loads(Path(path).read_text())
        mean, std = doc.get("mean"), doc.get("std")
        if mean is None or std is None:
            rows.append({"json": path, "note": "no mean/std keys", "keys": sorted(doc)[:8]})
            continue
        for name, dtype in DTYPES.items():
            dm, tm = digest(mean, dtype)
            ds, ts = digest(std, dtype)
            match = dm == want.get("force_mean_sha256") and ds == want.get("force_std_sha256")
            rows.append({"json": path, "rounding": name, "mean_sha256": dm[:16],
                         "std_sha256": ds[:16], "MATCHES_FINGERPRINT": match,
                         "channels": doc.get("channels"), "force_dim": doc.get("force_dim"),
                         "hist_len": doc.get("hist_len"), "scale": doc.get("scale"),
                         "n_values": len(mean), "mean_absmax": float(tm.abs().max()),
                         "std_min": float(ts.min())})
    for r in rows:
        print(json.dumps(r, sort_keys=True))
    hits = [r for r in rows if r.get("MATCHES_FINGERPRINT")]
    print(json.dumps({"candidates": len(cands), "matches": len(hits),
                      "match": None if not hits else {k: hits[0][k] for k in
                                                      ("json", "rounding", "channels")},
                      "ambiguous": bool(len(hits) > 1)}, sort_keys=True))
    return 0 if len(hits) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
