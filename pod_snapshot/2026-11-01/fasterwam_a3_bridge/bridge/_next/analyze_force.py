"""Contact-force, grasp and release metrics for the r3 C0 baseline.

Companion to `analyze_lift.py` (which owns lift / stable-follow / transport) and
`analyze_gripper.py` (which owns flip counts and the Safety sign rewrite). This
script adds only the two families the baseline table still needs:

  * force: per-step contact force is `max(pad_force_inner, pad_force_outer)` in
    N. Thresholds are declared here, not chosen after seeing the numbers:
    >5 N (light clamp) and >10 N (hard clamp), plus p50/p90/p99. 25 N is the
    Safety soft limit, so `frac_over_25n` is EVAL_PROTOCOL §6's
    `frac_steps_above_soft`.
  * release: `drop_events` counts, per cookie, the transitions from lifted
    (>5 mm above the reset height, same threshold as analyze_lift) back to not
    lifted. With `cookies_in_target == 0` every one of those releases happened
    outside the target box, i.e. it is a premature release / lost grasp.
    `open_cmd_widening_while_holding` counts the steps in which the *applied*
    gripper command widened by more than 0.02 units while a cookie was held;
    the sign convention is not assumed -- `gripper_cmd_vs_aperture_corr` is the
    Pearson correlation between that command and the measured aperture, so the
    direction of "open" is read off the log rather than guessed.

Rewrite accounting is read straight off the executed log: on C0 the loop calls
`safety_wrap.apply` only (`run_matrix.py:168`), so `changed` is the Safety
rewrite mask and the Wrapper is not in the action path at all -- its rewrite
ratio is 0 by construction, and this script asserts that by checking the
wrapper-stage column of the diagnostic log is byte-identical to the pre-safety
model action.

    /opt/venv/bin/python3 analyze_force.py \
        --glob '/workspace/a3_eval/r3_c0_diag_600/rep0/lora_C0_*.npz' \
        --out  /workspace/a3_eval/r3_c0_diag_600/force.json
"""
import argparse
import glob
import json

import numpy as np

LIGHT_N = 5.0
HARD_N = 10.0
SOFT_N = 25.0          # EVAL_PROTOCOL §6 `frac_steps_above_soft` threshold
LIFT_M = 0.005
OPEN_CMD_M = 0.02      # widening of the applied gripper command, in action units


def seed_of(path):
    return int(str(path).rsplit('_', 1)[1][:-len('.npz')])


def longest(mask):
    best = run = 0
    for x in mask:
        run = run + 1 if x else 0
        best = max(best, run)
    return int(best)


def score(path):
    m = np.load(path)
    pad = np.maximum(m["pad_force_inner"], m["pad_force_outer"])
    dz = m["cookie_xyz"][:, :, 2] - m["cookie_xyz0"][:, 2]
    lifted = dz > LIFT_M

    # grasp quality: lifted + both pads loaded + no slip against the TCP
    contact = (m["pad_force_inner"] > 2.0) & (m["pad_force_outer"] > 2.0)
    rel = m["cookie_xyz"] - m["eef_xyz"][:, None, :]
    slip = np.zeros(dz.shape, dtype=bool)
    slip[1:] = np.linalg.norm(np.diff(rel, axis=0), axis=2) < 0.002
    good = lifted & contact[:, None] & slip

    drops = 0
    for c in range(dz.shape[1]):
        x = lifted[:, c].astype(np.int8)
        drops += int(np.sum(np.diff(x) == -1))
    ever_lifted = int(np.sum(lifted.any(axis=0)))

    # "holding" is the physical state (something off the surface + both pads
    # loaded); the slip term is left out because a slip step is exactly what a
    # release attempt looks like.
    holding = lifted.any(axis=1) & contact
    cmd = m["applied_opening"]
    aperture = m["opening"]
    corr = float(np.corrcoef(cmd, aperture)[0, 1]) if cmd.std() > 1e-9 else float("nan")
    step_cmd = np.diff(cmd, prepend=cmd[0])
    # Direction of "open" is read off the log: the applied gripper command
    # co-varies with the measured aperture with the sign of `corr`.
    widening = (step_cmd > OPEN_CMD_M) if corr >= 0 else (step_cmd < -OPEN_CMD_M)
    narrowing = (step_cmd < -OPEN_CMD_M) if corr >= 0 else (step_cmd > OPEN_CMD_M)

    pre, post = m["pre_14d"], m["post_14d"]
    changed = np.any(np.abs(pre - post) > 1e-9, axis=1)
    row = {
        "seed": seed_of(path),
        "steps": int(pre.shape[0]),
        "pad_force_max_n": round(float(pad.max()), 3),
        "pad_force_p50_n": round(float(np.percentile(pad, 50)), 3),
        "pad_force_p90_n": round(float(np.percentile(pad, 90)), 3),
        "pad_force_p99_n": round(float(np.percentile(pad, 99)), 3),
        "steps_over_5n": int(np.sum(pad > LIGHT_N)),
        "frac_over_5n": round(float((pad > LIGHT_N).mean()), 4),
        "steps_over_10n": int(np.sum(pad > HARD_N)),
        "frac_over_10n": round(float((pad > HARD_N).mean()), 4),
        "steps_over_25n_soft_limit": int(np.sum(pad > SOFT_N)),
        "frac_steps_above_soft": round(float((pad > SOFT_N).mean()), 4),
        "hold_steps": int(np.sum(holding)),
        "hold_frac": round(float(holding.mean()), 4),
        "gripper_cmd_vs_aperture_corr": round(corr, 4),
        "open_cmd_events_while_holding": int(np.sum(widening & holding)),
        "close_cmd_events_while_holding": int(np.sum(narrowing & holding)),
        "wrist_force_max_n": round(float(np.linalg.norm(m["wrist_force"], axis=1).max()), 2),
        "contact_both_pads_steps": int(np.sum(contact)),
        "contact_both_pads_frac": round(float(contact.mean()), 4),
        "grasp_frames_frac": round(float(good.any(axis=1).mean()), 4),
        "stable_follow_steps": max(longest(good[:, c]) for c in range(good.shape[1])),
        "cookies_ever_lifted": ever_lifted,
        "premature_release_events": drops,
        "max_cookie_lift_mm": round(float(dz.max() * 1000.0), 1),
        "safety_rewritten_steps": int(np.sum(changed)),
        "safety_rewritten_frac": round(float(changed.mean()), 4),
        "safety_rewritten_gripper_dim": int(np.sum(np.abs(pre[:, 6] - post[:, 6]) > 1e-9)),
        "verbatim_pct": round(float(100.0 * (~changed).mean()), 2),
    }
    if "post_wrap_14d" in m.files:
        # Wrapper rewrite is measured against the wrapper's OWN input (`pre`,
        # the adapted model action). `post_14d` already contains the safety
        # layer, so diffing against it would report Safety edits as Wrapper
        # edits. On C0 `run_matrix.py` sets `post_w = pre.copy()`, so the two
        # arrays must be bitwise identical and the Wrapper rewrite ratio 0.
        pw = m["post_wrap_14d"]
        wc = np.any(np.abs(pw - pre) > 1e-9, axis=1)
        row["wrapper_rewritten_steps"] = int(np.sum(wc))
        row["wrapper_rewritten_frac"] = round(float(wc.mean()), 4)
        row["wrapper_stage_bitwise_identical_to_model_action"] = bool(np.array_equal(pw, pre))
        row["safety_diff_measured_against_wrapper_output"] = bool(
            np.array_equal(changed, np.any(np.abs(post - pw) > 1e-9, axis=1)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = [score(p) for p in sorted(glob.glob(args.glob))]
    text = json.dumps(rows, indent=1)
    print(text, flush=True)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
