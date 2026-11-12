"""Re-adjudicate already-recorded correction demos with a recovery-segment rule.

The per-clamp acceptance in replay_forces.py only looks at the grasp segments, so an
episode whose *recovery* still contains a force spike can pass it.  Here the recording is
split at the retry_log cut points and both halves are measured: the abnormal half must
actually be abnormal, the recovered half must stay inside the corpus envelope.

Read-only: it never rewrites the npz or summary.json, it only prints verdicts.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import replay_forces as R

ANOMALY_PAD_N = 1.6
ANOMALY_FRAMES = 3
HARD_CAP_N = 2.0          # corpus absolute maximum over 42681 expert frames is 1.540 N


def episode_rows(summary):
    for row in json.loads(Path(summary).read_text(encoding="utf-8")):
        yield row


def describe(pad, a, b):
    seg = pad[a:b]
    return {
        "slice": "%d:%d" % (a, b),
        "n": int(len(seg)),
        "peak": round(float(seg.max()), 3) if len(seg) else 0.0,
        "over_1p6": int(np.sum(seg > ANOMALY_PAD_N)),
        "over_2p0": int(np.sum(seg > HARD_CAP_N)),
    }


def adjudicate(out_dir, max_spikes=3):
    out = Path(out_dir)
    summary = out / "summary.json"
    if not summary.exists():
        print("no summary.json in", out)
        return []
    rows = list(episode_rows(summary))
    z = np.load(out / ("expert_correct_%d.npz" % rows[0]["seed"]))
    keys = sorted(z.files)
    print("npz_keys", ",".join(keys))
    verdicts = []
    for row in rows:
        seed = int(row["seed"])
        zp = out / ("expert_correct_%d.npz" % seed)
        if not zp.exists():
            print("seed=%d MISSING npz" % seed)
            continue
        z = np.load(zp)
        pad = np.maximum(z["pad_force_inner"], z["pad_force_outer"])
        phase = z["phase"]
        cuts = [int(r["at_step"]) for r in row.get("retry_log", [])]
        segs = []
        bounds = [0] + cuts + [len(pad)]
        for i in range(len(bounds) - 1):
            tag = "abnormal" if i < len(cuts) else "recovery"
            segs.append(dict(segments=describe(pad, bounds[i], bounds[i + 1]), kind=tag,
                             reason=(row["retry_log"][i]["anomaly_kind"]
                                     if i < len(cuts) else "-")))
        # where the recovery spikes are, with the neighbourhood, to separate a genuine
        # mid-recovery spike from a boundary artifact of the cut
        spikes = []
        if cuts:
            rec = pad[cuts[-1]:]
            for idx in np.flatnonzero(rec > HARD_CAP_N)[:max_spikes]:
                g = int(cuts[-1] + idx)
                lo, hi = max(0, g - 4), min(len(pad), g + 5)
                spikes.append({
                    "frame": g,
                    "frames_after_cut": g - cuts[-1],
                    "pad_window": [round(float(v), 3) for v in pad[lo:hi]],
                    "phase_window": [R.PHASE_NAMES[int(c)] for c in phase[lo:hi]],
                    "opening_window": [round(float(v), 3) for v in z["opening"][lo:hi]],
                })
        abnormal_ok = any(s["kind"] == "abnormal"
                          and s["segments"]["over_1p6"] >= ANOMALY_FRAMES
                          and s["segments"]["peak"] > ANOMALY_PAD_N for s in segs)
        rec_seg = next((s for s in reversed(segs) if s["kind"] == "recovery"), None)
        recovery_ok = rec_seg is not None and rec_seg["segments"]["over_2p0"] == 0 \
            and rec_seg["segments"]["peak"] <= ANOMALY_PAD_N
        clamps = row.get("clamps", [])
        clamp_ok = len(clamps) == 1 and bool(clamps[0]["accept"]) and \
            int(clamps[0].get("placed_upright", 0)) == 1
        single_grasp = len(clamps) == 1
        verdict = {
            "seed": seed,
            "variant": int(row.get("variant", -1)),
            "steps": int(row["steps"]),
            "cuts": cuts,
            "segments": segs,
            "recovery_spikes": spikes,
            "n_clamps": len(clamps),
            "loose_accept": bool(row.get("accept")),
            "abnormal_observed": bool(abnormal_ok),
            "recovery_clean": bool(recovery_ok),
            "grasp_valid": bool(clamp_ok),
            "single_grasp_segment": bool(single_grasp),
            "strict_accept": bool(abnormal_ok and recovery_ok and clamp_ok and single_grasp),
        }
        verdicts.append(verdict)
        print(json.dumps(verdict, default=float), flush=True)
    return verdicts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", nargs="+", required=True)
    args = ap.parse_args()
    tally = []
    for d in args.dirs:
        print("=== %s ===" % d, flush=True)
        vs = adjudicate(d)
        tally.append({
            "dir": d,
            "n": len(vs),
            "loose": sum(1 for v in vs if v["loose_accept"]),
            "strict": sum(1 for v in vs if v["strict_accept"]),
            "rejected_by_recovery": sum(1 for v in vs
                                        if v["loose_accept"] and not v["recovery_clean"]),
        })
    print("=== tally ===")
    for t in tally:
        print(json.dumps(t))


main()
