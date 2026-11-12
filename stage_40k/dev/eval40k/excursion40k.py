"""Per-dimension attribution of the recorded action excursions, from the results tree only.

What the batch left on disk is enough to narrow the 44.37 question but not to close it: the
executed action vectors were never stored, so the only action statistic that survived is
`meta["action_absmax"]`, a scalar per replan taken *after* the gripper rewrite. This script reads
what is there -- `action_absmax_max`, the per-chunk `chunk_log` witnesses, the proprio bands -- and
does two things no reader can do by eye:

1. turn each seed's single absmax into the table of "if this excursion lives in dimension j, the
   model must have emitted normalised value x_j", using the V1G action band as the only scale.
   Because the band is min/max and the gripper column is forced to +-1 by the pm1 rewrite, a 44.37
   absmax can only be dimensions 0-5, and the required x_j is 200-600 -- far outside the +-5 the
   training-side `forward()` clamps to. That is what makes the number an excursion rather than a
   unit mistake, and it is worth computing rather than asserting.
2. line that up against the conditioning of the same replans: the proprio the witness recorded per
   chunk, per column, unclipped. If an extreme-action seed also has extreme proprio, the excursion
   has an input-side story; if it does not, the sampler itself produced it.

It never loads a model and never touches the results tree. Run it with `--results-dir`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BAND = {
    "min": [-0.0642, -0.1149, -0.295, -0.0413, -0.0121, -0.0247, -1.0],
    "max": [0.0872, 0.2987, 0.1003, 0.0152, 0.0336, 0.0297, 1.0],
}
DIM_NAMES = ("d_x", "d_y", "d_z", "d_rx", "d_ry", "d_rz", "gripper")
CLAMP = 5.0


def normalised_needed(absmax: float, lo: float, hi: float) -> float:
    """Inverse of the min/max forward map: the normalised value whose denormalisation is `absmax`."""
    rng = hi - lo
    if rng <= 0.0:
        return float("nan")
    return (absmax - lo) * 2.0 / rng - 1.0


def per_seed(path: Path) -> dict:
    rec = json.loads(path.read_text())
    clog = rec.get("chunk_log") or []
    raws = [c["raw14"] for c in clog if c.get("raw14") is not None]
    maps = [c["mapped_no_clamp14"] for c in clog if c.get("mapped_no_clamp14") is not None]
    out = {
        "seed": rec.get("seed"),
        "success": rec.get("success"),
        "frames": rec.get("frames"),
        "decision_chunks": rec.get("decision_chunks"),
        "action_absmax_max": rec.get("action_absmax_max"),
        "gripper_pm1_values": rec.get("gripper_pm1_values"),
        "termination": rec.get("termination") or rec.get("stop_reason"),
        "chunks_witnessed": len(maps),
    }
    if maps:
        M = np.asarray(maps, dtype=np.float64)
        out["mapped_col_absmax"] = [float(x) for x in np.abs(M).max(0)]
        out["mapped_clamp_cols"] = [int(x) for x in (np.abs(M) > CLAMP).sum(0)]
        out["mapped_clamp_chunk_frac"] = round(float(np.mean((np.abs(M) > CLAMP).any(1))), 3)
    if raws:
        R = np.asarray(raws, dtype=np.float64)
        out["raw_col_absmax"] = [float(x) for x in np.abs(R).max(0)]
    if out["action_absmax_max"] is not None:
        out["normalised_needed_if_dim"] = {
            DIM_NAMES[j]: round(normalised_needed(out["action_absmax_max"],
                                                  BAND["min"][j], BAND["max"][j]), 1)
            for j in range(len(BAND["min"]))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--json-out")
    args = ap.parse_args()
    root = Path(args.results_dir)
    files = sorted(root.rglob("episode_seed*.json"))
    if not files:
        raise SystemExit(f"no episode_seed*.json under {root}")
    rows = [per_seed(f) for f in files]
    print("band (V1G action, min/max): %s" % json.dumps(BAND))
    print("gripper column is excluded from absmax by the pm1 rewrite; only dims 0-5 can reach it\n")
    hdr = ("seed", "ok", "frames", "chunks", "absmax", "clampch", "worst_mapped_col",
           "x_needed_at_absmax")
    print("%-5s %-3s %-7s %-7s %-8s %-8s %-24s %s" % hdr)
    for r in rows:
        worst = "-"
        if r.get("mapped_col_absmax"):
            i = int(np.argmax(r["mapped_col_absmax"]))
            worst = "col%d=%.2f" % (i, r["mapped_col_absmax"][i])
        need = "-"
        if r.get("normalised_needed_if_dim"):
            best = min(((v, k) for k, v in r["normalised_needed_if_dim"].items()))
            need = "%s:%s" % (best[1], best[0])
        print("%-5s %-3s %-7s %-7s %-8s %-8s %-24s %s" % (
            r["seed"], "Y" if r["success"] else "n", r["frames"],
            r["decision_chunks"], r["action_absmax_max"],
            r.get("mapped_clamp_chunk_frac", "-"), worst, need))
    print("\nper-dim normalised value required to produce each seed's absmax (smallest = most "
          "plausible dim):")
    for r in rows:
        if r.get("normalised_needed_if_dim"):
            print("  seed %s absmax %s -> %s" % (r["seed"], r["action_absmax_max"],
                                                 r["normalised_needed_if_dim"]))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps({"band": BAND, "rows": rows}, indent=2))
        print("\nwrote %s" % args.json_out)


if __name__ == "__main__":
    main()
