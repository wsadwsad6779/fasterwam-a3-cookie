"""Break the C0 gripper flips down by where they happen and how far raw is from 0.

`eval_c0_diag.py` stores two views of the same episode:

  * `<model>_<control>_<seed>.npz` -- executed steps: `pre_14d` (after pm1) etc.
  * `diag_..._r<rep>.npz`         -- every predicted chunk: `raw_gripper`
                                     (before binarization, model sign convention)
                                     and `bin_gripper` (after pm1).

A flip counted on `pre_14d` mixes three different things: a new chunk deciding
the opposite of the last observed step, a within-chunk sign change that the
10-step execution window happened to include, and a change the Safety layer
made. This script separates them and reports how close the continuous gripper
was to the 0 threshold at each flip, which is what decides whether pm1 is
turning a weak signal into a hard ±1.

    python3 analyze_gripper.py --seed 21 \
        --run /workspace/a3_eval/r3_c0_diag_600/rep0/lora_C0_21.npz \
        --diag /workspace/a3_eval/r3_c0_diag_600/rep0/diag_lora_C0_21_r0.npz \
        --out /workspace/a3_eval/r3_c0_diag_600/gripper_21.json
"""
import argparse
import json
from pathlib import Path

import numpy as np


def sign(x):
    return np.where(np.asarray(x) >= 0, 1, -1)


def flips(a):
    s = sign(a)
    return int(np.sum(s[1:] != s[:-1]))


def positions(chunk_idx):
    """Index of each executed step inside its chunk (0-based, per chunk)."""
    counts, pos = {}, np.zeros(len(chunk_idx), dtype=int)
    for t, c in enumerate(chunk_idx):
        pos[t] = counts.get(int(c), 0)
        counts[int(c)] = pos[t] + 1
    return pos, counts


def quantiles(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {}
    q = np.quantile(np.abs(x), [0.5, 0.9, 0.99])
    return {"abs_median": round(float(q[0]), 4),
            "abs_p90": round(float(q[1]), 4),
            "abs_p99": round(float(q[2]), 4),
            "frac_abs_lt_0.05": round(float(np.mean(np.abs(x) < 0.05)), 4),
            "frac_abs_lt_0.10": round(float(np.mean(np.abs(x) < 0.10)), 4),
            "frac_abs_lt_0.25": round(float(np.mean(np.abs(x) < 0.25)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--diag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--contact-force", type=float, default=5.0,
                    help="pad force (N) above which a step counts as in contact")
    ap.add_argument("--hold-opening", type=float, default=0.6,
                    help="opening below which the fingers are closed enough to be holding")
    args = ap.parse_args()

    with np.load(args.run, allow_pickle=False) as m:
        pre = m["pre_14d"][:, 6]
        post = m["post_14d"][:, 6]
        chunk_idx = m["chunk_idx"]
        opening = m["opening"]
        applied_opening = m["applied_opening"]
        force = m["pad_force_inner"]
        changed = m["changed"]
    with np.load(args.diag, allow_pickle=False) as d:
        raw = d["raw_gripper"]        # [n_chunks, 32]
        binned = d["bin_gripper"]     # [n_chunks, 32]

    pos, counts = positions(chunk_idx)
    n_chunks = len(counts)
    n_steps = len(pre)
    # raw model-space value for the step actually executed at each t
    raw_exec = np.asarray([raw[int(c), int(p)] for c, p in zip(chunk_idx, pos)])

    executed = sign(pre)
    boundary_mask = np.zeros(n_steps, dtype=bool)
    boundary_mask[1:] = chunk_idx[1:] != chunk_idx[:-1]
    exec_flips = executed[1:] != executed[:-1]
    within = exec_flips & ~boundary_mask[1:]

    # within-chunk sign changes over all 32 predicted steps
    within_chunk_all = int(sum(flips(row) for row in raw))
    # seam between the last executed step of a chunk and the first of the next
    seam = 0
    first_step = {}
    for t, c in enumerate(chunk_idx):
        first_step.setdefault(int(c), t)
    ids = sorted(first_step)
    for a, b in zip(ids[:-1], ids[1:]):
        last_pos = int(pos[first_step[a] + counts[a] - 1])
        seam += int(sign(raw[b, 0]) != sign(raw[a, last_pos]))

    contact = np.asarray(force) > args.contact_force
    holding = contact & (np.asarray(opening) < args.hold_opening)
    safety = np.asarray(changed)

    # within-chunk sign changes over the steps actually executed from each chunk:
    # the last executed position of chunk c is at first_step[c] + counts[c] - 1.
    last_pos = {c: int(pos[first_step[c] + counts[c] - 1]) for c in counts}
    chunk_of_row = {int(c): i for i, c in enumerate(ids)}
    within_chunk_exec = int(sum(
        flips(raw[chunk_of_row[c], :last_pos[c] + 1]) for c in ids))

    out = {
        "seed": args.seed,
        "steps": n_steps,
        "chunks": n_chunks,
        "executed": {
            "flips_pre_14d": flips(pre),
            "flips_post_safety": flips(post),
            "at_chunk_boundary": int(np.sum(exec_flips & boundary_mask[1:])),
            "within_chunk": int(within.sum()),
            "safety_rewritten_steps": int(safety.sum()),
            "safety_pct": round(100.0 * float(safety.mean()), 2),
        },
        "predicted_chunks": {
            "within_chunk_flips_all_32": within_chunk_all,
            "within_chunk_flips_executed_prefix": within_chunk_exec,
            "seam_flips_between_chunks": seam,
            "raw_equals_bin_everywhere": bool(np.array_equal(sign(raw), binned)),
        },
        "threshold_distance": {
            "all_32_steps": quantiles(raw.ravel()),
            "executed_steps": quantiles(raw_exec),
            "executed_flip_steps": quantiles(raw_exec[1:][exec_flips]),
        },
        "context": {
            "in_contact_steps": int(contact.sum()),
            "holding_steps": int(holding.sum()),
            "flips_while_holding": int(np.sum(exec_flips & holding[1:])),
            "flips_while_contact_not_holding": int(np.sum(exec_flips & contact[1:] & ~holding[1:])),
            "flips_free_space": int(np.sum(exec_flips & ~contact[1:])),
            "safety_steps_that_changed_sign": int(np.sum(safety & (sign(pre) != sign(post)))),
        },
        "raw_stats": {
            "raw_min": round(float(raw.min()), 4),
            "raw_max": round(float(raw.max()), 4),
            "raw_mean": round(float(raw.mean()), 4),
            "frac_positive_executed": round(float(np.mean(raw_exec >= 0)), 4),
        },
    }

    print(json.dumps(out, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
