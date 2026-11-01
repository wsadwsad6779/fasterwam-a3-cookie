"""C-3 (the gripper closed and held nothing -> replanned single-cookie grasp) re-scored.

docs_task2 6.2 lists five correction families. The C-1 rule used by adjudicate_correct.py
-- the abnormal half must exceed 1.6 N for at least 3 frames -- rejects by construction
every episode whose closure met no resistance at all, so the episodes inside the
demos_correct_* directories that the driver itself labels empty_grasp / stuck_no_force
have never been scored under their own criteria. This tool does that, read-only.

The abnormal half is judged on what the scripted expert reported and on what the cookies
actually did, not on a force level: its reason string carries the experts own contact
count (contacts=[0.0, 0.0]) and the failed clamp must show no cookie in the gripper column
while carried and no cookie raised to carry height. The recovery half must stay inside the
corpus envelope (the absolute maximum over 42681 expert frames is 1.540 N) and must end in
a single-cookie grasp whose closure hold is in [0.3, 1.6] N, released upright inside the
bin -- that hold window is the C-3 success criterion of 6.2.

true_empty separates a closure that literally touched nothing (peak < 0.05 N while the
fingers ran down to the commanded close opening through air) from the stuck_no_force ones
(the pads pressed the stack with 0.68-2.28 N and still nothing came away held). 6.2 words
the trigger as "max(pad) < 0.05 N and the opening does not stall", which is the former.

Read-only with respect to the recordings. --write drops an adjudication_c3.json next to
each directory and refuses to overwrite one that already exists.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import replay_forces as R

HOLD_MIN_N = 0.3
ANOMALY_PAD_N = 1.6
HARD_CAP_N = 2.0          # corpus absolute maximum over 42681 expert frames is 1.540 N
EMPTY_PEAK_N = 0.05       # 6.2: the C-3 trigger is max(pad) < 0.05 N
CLOSED_THROUGH_AIR = 0.10  # opening that counts as the fingers having run fully down
CARRY_LIFT_MM = 60.0      # clamp_stats accept needs >= 60 mm, so below it = never carried
NO_CONTACT = "contacts=[0.0, 0.0]"


def describe(pad, a, b):
    seg = pad[a:b]
    return {
        "slice": "%d:%d" % (a, b),
        "n": int(len(seg)),
        "peak": round(float(seg.max()), 3) if len(seg) else 0.0,
        "over_1p6": int(np.sum(seg > ANOMALY_PAD_N)),
        "over_2p0": int(np.sum(seg > HARD_CAP_N)),
    }


def adjudicate(out_dir):
    out = Path(out_dir)
    summary = out / "summary.json"
    if not summary.exists():
        print("no summary.json in", out)
        return []
    rows = json.loads(summary.read_text(encoding="utf-8"))
    episodes = []
    for row in rows:
        log = row.get("retry_log") or []
        kinds = [str(e.get("anomaly_kind")) for e in log]
        if not log or not any(k in ("empty_grasp", "stuck_no_force") for k in kinds):
            continue
        seed = int(row["seed"])
        zp = out / ("expert_correct_%d.npz" % seed)
        if not zp.exists():
            print("seed=%d MISSING npz" % seed)
            continue
        z = np.load(zp)
        pad = np.maximum(z["pad_force_inner"], z["pad_force_outer"])
        ph = [R.PHASE_NAMES[int(x)] for x in z["phase"]]
        xyz = np.asarray(z["cookie_xyz"])
        eef = np.asarray(z["eef_xyz"])
        z0 = np.asarray(z["cookie_xyz0"])
        op = np.asarray(z["opening"])
        cut = int(log[0]["at_step"])
        rec = {
            "seed": seed, "file": zp.name, "variant": int(row["variant"]),
            "noise_m": row["noise_m"], "steps": int(row["steps"]), "stop": row["stop"],
            "cut": cut, "anomaly_kinds": kinds,
            "anomaly_reason": str(log[0].get("reason", "")),
            "expert_reports_no_contact": NO_CONTACT in str(log[0].get("reason", "")),
            "abnormal": describe(pad, 0, cut), "recovery": describe(pad, cut, len(pad)),
        }
        rec["true_empty"] = bool(rec["abnormal"]["peak"] < EMPTY_PEAK_N)

        segs = R.segments(np.asarray(ph[:cut]))
        rec["abnormal_clamp"] = None
        rec["held_nothing"] = False
        if segs:
            a, b = segs[-1]
            st = R.clamp_stats(a, b, ph[:cut], pad[:cut], xyz[:cut], eef[:cut], z0, op[:cut], None)
            keys = ("phases", "closure_peak_pad_n", "carry_peak_pad_n", "max_cookies_at_grip",
                    "min_opening", "max_opening", "carried_lift_mm", "carried_distance_mm")
            rec["abnormal_clamp"] = {k: st[k] for k in keys}
            rec["held_nothing"] = bool(st["max_cookies_at_grip"] == 0
                                       and st["carried_lift_mm"] < CARRY_LIFT_MM)
            rec["closed_through_air"] = bool(rec["true_empty"]
                                             and st["min_opening"] <= CLOSED_THROUGH_AIR)
        else:
            rec["closed_through_air"] = False

        rec["recovery_clean"] = bool(rec["recovery"]["over_2p0"] == 0
                                     and rec["recovery"]["peak"] <= ANOMALY_PAD_N)
        rec["recovery_clamp"] = None
        rec["grasp_accept"] = False
        rec["hold_in_range"] = False
        segs = R.segments(np.asarray(ph[cut:]))
        if segs:
            a, b = segs[-1]
            st = R.clamp_stats(a, b, ph[cut:], pad[cut:], xyz[cut:], eef[cut:], z0, op[cut:], None)
            keys = ("phases", "closure_peak_pad_n", "hold_mean_pad_n", "carry_peak_pad_n",
                    "max_cookies_at_grip", "carried_lift_mm", "placed_in_bin_xy",
                    "placed_upright", "accept")
            rec["recovery_clamp"] = {k: st[k] for k in keys}
            rec["grasp_accept"] = bool(st["accept"])
            hold = st["closure_peak_pad_n"]
            rec["hold_in_range"] = bool(hold is not None and HOLD_MIN_N <= hold <= ANOMALY_PAD_N)

        rec["strict_accept"] = bool(rec["expert_reports_no_contact"] and rec["held_nothing"]
                                    and rec["recovery_clean"] and rec["grasp_accept"]
                                    and rec["hold_in_range"])
        episodes.append(rec)
        brief = ("seed", "variant", "anomaly_kinds", "true_empty", "closed_through_air",
                "held_nothing", "recovery_clean", "grasp_accept", "hold_in_range",
                "strict_accept")
        print(json.dumps({k: rec[k] for k in brief}))
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    note = ("C-3 empty/non-holding closure corrections, re-scored on the retry_log cut. "
            "abnormal half = a clamp that held nothing (expert reason %s, max_cookies_at_grip 0, "
            "carried_lift_mm < %g); recovery half = inside the corpus envelope and a single-cookie "
            "grasp with closure hold in [%g, %g] N released upright in the bin. true_empty marks the "
            "closures that touched nothing at all (< %g N), the ones 6.2 words literally." % (
                NO_CONTACT, CARRY_LIFT_MM, HOLD_MIN_N, ANOMALY_PAD_N, EMPTY_PEAK_N))
    for d in args.dirs:
        episodes = adjudicate(d)
        tally = {"dir": d, "n": len(episodes)}
        for key in ("true_empty", "closed_through_air", "expert_reports_no_contact",
                    "held_nothing", "recovery_clean", "grasp_accept", "hold_in_range",
                    "strict_accept"):
            tally[key] = int(sum(1 for e in episodes if e.get(key)))
        print("TALLY", json.dumps(tally))
        if args.write:
            target = Path(d) / "adjudication_c3.json"
            if target.exists():
                print("REFUSING to overwrite", str(target))
            else:
                blob = json.dumps({"note": note,
                                   "thresholds": {"hold_min_n": HOLD_MIN_N,
                                                  "anomaly_pad_n": ANOMALY_PAD_N,
                                                  "hard_cap_n": HARD_CAP_N,
                                                  "empty_peak_n": EMPTY_PEAK_N,
                                                  "carry_lift_mm": CARRY_LIFT_MM},
                                   "tally": tally, "episodes": episodes}, indent=2)
                target.write_text(blob, encoding="utf-8")
                print("WROTE", str(target), "bytes", len(blob),
                      "sha", hashlib.sha256(blob.encode()).hexdigest()[:16])


if __name__ == "__main__":
    main()
