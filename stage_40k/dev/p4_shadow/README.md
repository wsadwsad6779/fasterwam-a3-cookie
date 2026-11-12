# p4_shadow — the observer, and what its thresholds actually cost on real force data

Shadow Mode is read-only by construction: `ShadowMonitor` has no code path that returns an
action, the rollout never hands it one, and `finalize()` states `actions_mutated: 0`. The verdict
(`proceed` / `hold` / `stop`) is logged so that the same thresholds can be turned on later with a
measured false-positive rate instead of a guess. This file is where that rate lives.

```sh
cd /workspace/_v1_stage/dev/p4_shadow
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python shadow_selftest.py --json out/shadow_selftest.json
PYTHONDONTWRITEBYTECODE=1 nice -n 19 python fp_audit.py --out out/fp_audit.json
```

* `shadow_selftest.py` — unit behaviour on synthetic streams (does each rule fire, does each
  clean case stay clean, is the log complete, is coverage honest).
* `fp_audit.py` — the same rules against the frozen expert corpus
  (`/workspace/a3_data/a3_fw_force/arrays.npz`, 100 episodes / 42,681 frames), using the measured
  future as the prediction. That oracle is not a model result: it is the only way to ask "how
  often does the detector fire when the prediction is exactly right?", which is the definition of
  its false-positive floor. Decisions are counted only where the whole 32-step future is inside
  the episode, so no window is padded with invented values.

## Results, measured on the pod 2026-11-10 (CPU only, no GPU touched)

`shadow_selftest.py` — **31/31 checks pass** (`out/shadow_selftest.json`,
`n_checks: 31, n_passed: 31, failed: []`). The checks that matter for "default off is not a
promise but a property": the monitor returns a verdict document and never an action, the predicted
and measured arrays are byte-identical before and after observation, `mode="enforce"` is *refused by
this class* so the flag cannot drift on its own, the JSONL log has exactly one line per decision, and
a calibration with a zero or NaN `std` is refused rather than silently producing infinite z.

`fp_audit.py` saw 3,998 decisions, 192 predicted values per decision (32 steps x 6 channels).

**Off-band rule** (`z = |x - mean| / std` from `force-calib-v1`, fires if any value in the window
exceeds `z_band`):

| z_band | per-value rate | **per-decision rate** | iid guess | Gaussian per-value |
| --- | --- | --- | --- | --- |
| 3 | 0.01039 | 0.1576 | 0.8655 | 2.7e-03 |
| 4 (shipped) | 0.00754 | **0.1273** | 0.7663 | 6.3e-05 |
| 5 | 0.00168 | 0.0745 | 0.2758 | 5.7e-07 |
| 6 | 0.00034 | 0.0218 | 0.0632 | 2.0e-09 |
| 8 | 0.0 | 0.0 | 0.0 | 1.3e-15 |

Three things follow from that table, and all of them are awkward enough to be worth saying:

1. **At the shipped `z_band = 4.0`, a guard would have held 12.7% of replans even with a perfect
   predictor.** Enforcement must not be switched on at that threshold: one decision in eight is
   not a safety signal, it is the threshold sitting inside the signal's own tail.
2. Force is nowhere near Gaussian. The empirical per-value rate at 4 sigma is 0.75%, about 120x
   the Gaussian 6.3e-5 — because `std` is a whole-corpus constant and the same channel carries
   both the no-contact baseline and full contact, which makes the tails heavy by construction.
3. Do not multiply per-value rates to get a per-decision rate. The 192 values in a window are
   strongly correlated (one excursion is one excursion), so the iid prediction of 77% is wrong by
   6x against the 12.7% actually measured. The measured column is the only one that may be quoted.

Per-channel `|z|` (p50 / p99 / p99.9 / max) explains where the tail sits:
`L_wrist_fx` 0.14/4.73/4.91/4.98, `L_wrist_fy` 0.44/3.73/5.67/5.70, `L_wrist_fz` 0.40/4.60/6.30/7.43,
`L_pad_inner` 1.03/1.12/1.13/1.23, `L_pad_outer` 1.03/1.12/1.14/1.18, `L_gripper_act` 1.00/1.74/1.89/1.89.
The three wrist force components are the only ones that can trip the band; the pad channels are
two-state sensors whose `|z|` never leaves 1.3, so they contribute 2 of 6 channels and no
detections. A single max-over-window rule therefore really means "is this wrist reading a contact
event", and a threshold in the middle of the wrist tail is a threshold on contact, not on faults.

**Disagrees-with-sensor** (`disagree_z = 4.0` shipped) measured against a hold-last predictor,
which is the scale reference rather than a false-positive rate because its errors are genuine:
per-decision max error is 1.88 sigma at p50, 4.79 at p90, 6.91 at p99, 8.36 at max; it would fire
on 12.7% of decisions at 4 sigma, 5.4% at 6, 0.08% at 8. Read together with the band table,
6 sigma is the first threshold that separates "different from the last reading" from "inside the
signal's normal range".

**Spike and stuck rules: zero false positives on real data.** Across all 42,681 frames the largest
one-frame change is 3.46 sigma, so `spike_z = 6.0` never fires — not once, and not on a closing
gripper either. The longest run of bit-identical consecutive 6-channel readings is 1, so
`stuck_frames = 5` never fires either. Caveats worth keeping: 20 Hz sampling spreads a grasp onset
over several frames, and a jerky learned policy could produce genuine large deltas, so "0 on
expert data" is a statement about expert data, not a proof the rule is safe for any policy.

**Cross-check, so the sweep is not a different rule.** `ShadowMonitor` itself was then run at its
shipped defaults over 20 episodes with the same oracle feed: 801 decisions, 104 off-band
decisions, `verdict_counts {proceed: 697, hold: 104, stop: 0}` — 104 analytic vs 104 counted, and
`fp_audit.py` exits non-zero if those two ever disagree. The same run is also the strongest
evidence that the alignment machinery works on real-length streams: 25,632 prediction steps
issued, 25,632 aligned (coverage 1.0), MAE exactly 0.0 on all six channels.

## What to do with this before any enforcement

`z_band = 4.0` on a max-over-32x6-window is not a usable guard threshold; the options the numbers
support are to raise it to ~6 (2.2% floor), or to make the rule per-step and per-channel with a
channel-specific threshold from the tail table above, or to require two consecutive decisions
before a `hold`. Whichever is chosen has to be re-measured with `fp_audit.py` — that script is the
gate, not this document.
