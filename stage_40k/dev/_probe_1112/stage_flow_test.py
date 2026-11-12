"""The eval harness tested against numbers you can write down before running it.

Item 1 of the current brief says to test the evaluation *flow* before spending GPU time on
it, and a flow test that only proves "the script did not crash" is worth nothing. So every
check here compares the harness output against an expectation computed a different way --
by hand, from the frozen contract, or from an explicit second loop -- and the file exits
nonzero if any of them disagrees. It runs on CPU in a few seconds: no model, no simulator,
no merge artifact, no checkpoint.

What is covered:

* the force-history padding rule (`flow` snapshot vs a reference window sliced directly out
  of the measured frame list, the same way `A3VideoDataset.force_windows` slices it);
* the reason `policy.observe()` must see *every* step: a stream advanced once per replan
  produces a history that is provably not the training window, asserted rather than stated;
* an exact force MAE, in two worlds. The stub predicts persistence + offset, so in a constant
  force world its per-channel MAE must be *exactly* |offset| and its persistence baseline
  exactly 0; in a ramp world the error is `offset + force(t) - force(f)`, and every reported
  MAE (overall, per horizon bucket, per contact subset) must equal the value computed from the
  fixture formula without the harness;
* non-imputation: steps whose future never happened are skipped, so the scored count must
  equal an independently summed `min(horizon, last_frame - issue_frame)` and coverage < 1;
* the two reference baselines (persistence, zero) recomputed from the measured array;
* success-rate accounting and the Wilson interval, including the k=0 and k=n corners;
* the contract gates refusing a tampered calibration, a wrong-width force vector and a NaN;
* the diffusion seed advancing across replans: `infer_action` re-seeds its generators on every
  call, so a constant seed hands all ~60 chunks of an episode the same initial noise, and
  `noise_seed` is what stops that. The full 20 seed x 60 chunk table is hashed and dumped, because
  it is the thing the eight nodes have to agree on later;
* the same property seen from the loop instead of from the function: `rollout40k.run_episode`
  driven by a stand-in model that records the `seed=` it is handed, so the assertion is on what
  `infer_action` really received at each of the 60 replans -- and the regression the fix replaced
  is caught by name, by replaying it.
* the text conditioning: the one inference-only key taken from the deployment config, the
  instruction read off a synthetic corpus instead of chosen, every way the corpus can fail to
  describe one prompt, the bit-exact comparison against a stand-in encoder, the real frozen corpus
  on the pod, and the source ordering that makes all of it the path the batch evaluator takes.

    cd /workspace/_v1_stage/dev/eval40k
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python flow_test.py
    PYTHONDONTWRITEBYTECODE=1 nice -n 19 python flow_test.py --json ../out40k/flow_test.json

The `FixtureEnv` at the bottom is the piece the driver imports for `--backend fixture`; it
is a test fixture, and its "successes" describe the fixture, not any policy.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from force_metrics import ForceErrorAccount  # noqa: E402
from force_stream import (CHANNELS, FORCE_OBS_KEY, HIST_LEN, ForceStream,  # noqa: E402
                          ForceContractError, check_calib, check_node_calibration,
                          select_channels)
from policy40k import ForcePolicy, StubPolicy, noise_seed  # noqa: E402
from rollout40k import run_episode  # noqa: E402

# --- fixture world -------------------------------------------------------------
#
# Force is a deterministic function of the frame index so the expected history and the
# expected error can both be written down before anything runs. Two worlds:
#
#   "ramp"  -- at rest until frame CONTACT_ON, then a per-channel linear ramp. Gives a real
#              subset of contact steps, a non-trivial persistence baseline, and an error that
#              grows with the horizon, so bucketing is exercised.
#   "const" -- every frame carries the same nonzero force. In this world a
#              persistence-plus-offset prediction has an error of exactly |offset| on every
#              channel and a persistence baseline of exactly zero, which is the crisp
#              exactness assertion for the metric code.
CONTACT_ON = 24
RAMP_N_PER_FRAME = 0.7
CHANNEL_SLOPE = (1.0, -0.5, 0.25, 2.0, -1.5, 0.125)
CONST_N = np.array([1.7, -0.9, 0.4, 3.1, -2.3, 0.2], dtype=np.float64)
N_FORCE_CHANNELS = 18
PACK_DECIMALS = 5               # ForceErrorAccount rounds every reported value to 5 places


def fixture_force(frame: int, mode: str = "ramp") -> np.ndarray:
    """The frozen six channels at frame `frame`, in newtons."""
    if mode == "const":
        return CONST_N.copy()
    ramp = 0.0 if frame < CONTACT_ON else float(frame - CONTACT_ON) * RAMP_N_PER_FRAME
    return np.array([slope * ramp for slope in CHANNEL_SLOPE], dtype=np.float64)


def force18(frame: int, mode: str = "ramp") -> np.ndarray:
    """The full 18-wide vector the A3 env would emit at that frame: six real channels, zeros elsewhere."""
    vec = np.zeros(N_FORCE_CHANNELS, dtype=np.float64)
    vec[list(CHANNELS)] = fixture_force(frame, mode)
    return vec


def episode_length(seed: int) -> int:
    """Frames until the fixture terminates -- deliberately not a multiple of the replan."""
    return 3 + (int(seed) * 7) % 47


def succeeds(seed: int) -> bool:
    """The fixture's own success rule: short episodes are the ones that 'worked'."""
    return episode_length(seed) % 3 == 0


class FixtureEnv:
    """A stand-in environment with a known force trace and a known length.

    It emits the same 18-wide `observation.force` layout the A3 env does -- six frozen
    channels at the contract indices, everything else zero -- so `select_channels()` and the
    whole width check are exercised by the real code path, not a mock of it.
    """

    def __init__(self, seed: int = 0, frames: int = 80, force_mode: str = "ramp"):
        self.frame = -1
        self.seed = int(seed)
        self.force_mode = force_mode
        self.length = episode_length(seed)
        self.frames_cap = int(frames)
        self.steps_taken = 0

    def _obs(self):
        f = max(self.frame, 0)
        return {FORCE_OBS_KEY: force18(f, self.force_mode), "frame": f}

    def reset(self, seed=None):
        if seed is not None:
            self.seed = int(seed)
            self.length = episode_length(self.seed)
        self.frame = 0
        self.steps_taken = 0
        return self._obs()

    def step(self, action):
        self.frame += 1
        self.steps_taken += 1
        done = self.frame >= self.length
        truncated = False
        if self.frames_cap and self.frame >= self.frames_cap:
            done, truncated = True, True
        info = {"success": bool(done and not truncated and succeeds(self.seed)),
                "cookies_in_target": 1 if (done and succeeds(self.seed)) else 0}
        return self._obs(), 0.0, bool(done and not truncated), bool(truncated), info

    def close(self):
        pass


def reference_window(measured, frame, hist_len=HIST_LEN):
    """The dataset's own window, sliced from the measured list: rows s-(L-1) .. s, left-padded."""
    want = np.zeros((hist_len, len(CHANNELS)), dtype=np.float64)
    lo = max(0, frame - hist_len + 1)
    rows = [np.asarray(measured[i], dtype=np.float64) for i in range(lo, frame + 1)]
    if rows:
        want[hist_len - len(rows):] = np.stack(rows)
    mask = np.zeros(hist_len, dtype=bool)
    mask[hist_len - len(rows):] = True
    return want, mask


def rejections(fn, *args, **kw):
    """Return the exception message if `fn` raises ForceContractError, else None."""
    try:
        fn(*args, **kw)
    except ForceContractError as e:
        return str(e)
    return None


# --- checks --------------------------------------------------------------------

def check_padding(results):
    """Run one fixture episode and compare the stream to the reference window at every frame."""
    env = FixtureEnv(seed=11)
    policy = StubPolicy()
    obs = env.reset()
    policy.start_episode(obs, 11)
    measured = [select_channels(obs[FORCE_OBS_KEY])]
    policy.observe(obs, 0)
    bad = []
    for frame in range(0, min(env.length, 40)):
        got_hist, got_mask = policy.stream.snapshot()
        ref, ref_mask = reference_window(measured, frame)
        if not np.allclose(got_hist, ref, atol=1e-6):
            bad.append(frame)
        if not np.array_equal(got_mask, ref_mask):
            bad.append(("mask", frame))
        if got_mask.sum() != min(frame + 1, HIST_LEN):
            bad.append(("count", frame))
        obs, _r, _t, _tr, _i = env.step(np.zeros(14))
        measured.append(select_channels(obs[FORCE_OBS_KEY]))
        policy.observe(obs, frame + 1)
    first = [0, 1, 2, 3]
    hand = []
    for frame in first:
        hist, mask = reference_window(measured, frame)
        hand.append({"frame": frame, "true_rows": int(mask.sum()),
                     "last_row_newtons": [round(float(v), 4) for v in hist[-1]],
                     "padded_rows": int(HIST_LEN - mask.sum())})
    results["padding"] = {
        "frames_compared": len(range(0, min(env.length, 40))),
        "mismatched_frames": bad,
        "first_four_frames_hand_expectation": hand,
        "left_padded_not_right": bool(all(h["true_rows"] == h["frame"] + 1 for h in hand)),
    }
    results["checks"].append(("force history matches the dataset window at every frame",
                              bad == []))
    results["checks"].append(("the first four frames are left-padded with mask true only on "
                              "real rows", all(h["true_rows"] == h["frame"] + 1 for h in hand)))
    return measured


def check_time_base_guard(results, n_frames=41):
    """Show the trap the `observe()` assertion exists to catch, and that it does catch it."""
    raw = [force18(f) for f in range(n_frames)]
    measured = [select_channels(v) for v in raw]
    ref, ref_mask = reference_window(measured, n_frames - 1)

    consec = ForceStream()
    for vec in raw:
        consec.push(vec)
    matches = bool(np.allclose(consec.snapshot()[0], ref, atol=1e-6))

    skipped = ForceStream()
    for frame in (0, 10, 20, 30, 40):          # the mistake: one push per replan
        skipped.push(raw[frame])
    sk_hist, sk_mask = skipped.snapshot()
    diverges = not np.allclose(sk_hist, ref, atol=1e-3)

    pol = object.__new__(ForcePolicy)          # the guard is 4 lines; test them directly
    pol.stream = ForceStream()
    pol.stream.push(raw[0])
    raised = None
    try:
        pol.observe({"observation.force": raw[1]}, frame=5)
    except RuntimeError as e:
        raised = str(e)

    results["time_base"] = {
        "reference_frame": n_frames - 1,
        "consecutive_pushes_match_reference_window": matches,
        "replan_spaced_pushes_match_reference_window": bool(np.allclose(sk_hist, ref,
                                                                        atol=1e-3)),
        "replan_spaced_history_rows": np.round(sk_hist, 3).tolist(),
        "reference_window_rows": np.round(ref, 3).tolist(),
        "mask_after_skipping_true_rows": int(sk_mask.sum()),
        "mask_expected_true_rows": int(ref_mask.sum()),
        "observe_guard_message": raised,
    }
    results["checks"].append(("consecutive pushes reproduce the training window", matches))
    results["checks"].append(("a replan-spaced force history is NOT the training window "
                              "(so observe() must see every step)", diverges))
    results["checks"].append(("ForcePolicy.observe() refuses a mismatched frame",
                              raised is not None))


def _round5(v):
    return np.round(np.asarray(v, dtype=np.float64), PACK_DECIMALS)


def _run_world(mode, seeds, offset, horizon=32):
    """Roll the fixture in one force world and score it with the real account."""
    account = ForceErrorAccount(horizon=horizon)
    records, runs = [], []
    for seed in seeds:
        env = FixtureEnv(seed=seed, force_mode=mode)
        policy = StubPolicy(action_horizon=horizon, pred_mode="offset", offset=offset)
        rec, measured, chunks = run_episode(
            env, policy, seed=seed, reset=lambda e, s: e.reset(seed=s), max_steps=600, replan=10)
        for c in chunks:
            account.add_chunk(c["frame"], c["pred_force"], measured)
        records.append(rec)
        runs.append((seed, chunks))
    return account.finalize(), records, runs


def _expect(mode, seeds, offset, horizon=32):
    """The stub's own error written down from the fixture formula, without the harness.

    The stub predicts `force(issue_frame) + offset` for every future step, so a scored step's
    error is `offset + force(t) - force(f)`. That needs only the frame numbers, and in the
    constant-force world it collapses to exactly |offset| -- which is the crisp assertion.
    The number of frames an episode reaches is `episode_length(seed)`, again not read from the
    run, so the scored/unscored split is predicted rather than counted twice.
    """
    err, hold, zero, per_bucket = [], [], [], {}
    contact = scored = issued = 0
    for seed in seeds:
        n_frames = episode_length(seed)               # measured indices run 0 .. n_frames
        for t in range(0, n_frames, 10):              # one chunk per replan boundary
            issued += horizon
            base = fixture_force(t, mode)
            for j in range(horizon):
                f = t + 1 + j
                if f > n_frames:
                    continue
                truth = fixture_force(f, mode)
                scored += 1
                e = np.abs(offset + base - truth)
                err.append(e)
                hold.append(np.abs(base - truth))
                zero.append(np.abs(truth))
                if np.max(np.abs(truth)) >= 1.0:
                    contact += 1
                for b in ((0, 8), (8, 16), (16, 32)):
                    if b[0] <= j < b[1]:
                        per_bucket.setdefault(b, []).append(e)
    return {"mae": np.mean(np.stack(err), axis=0),
            "hold": np.mean(np.stack(hold), axis=0),
            "zero": np.mean(np.stack(zero), axis=0),
            "bucket_mae": {f"{a}-{b}": np.mean(np.stack(v), axis=0)
                           for (a, b), v in per_bucket.items()},
            "contact": contact, "scored": scored, "issued": issued,
            "chunks": issued // horizon}


def check_metrics(results, offset=0.5, horizon=32):
    """Score both fixture worlds and demand the account reproduce numbers computed elsewhere."""
    seeds = [11, 21, 22, 23, 24, 31, 32, 33, 41, 42, 51, 52]
    out = {}
    for mode in ("const", "ramp"):
        fin, records, runs = _run_world(mode, seeds, offset, horizon)
        exp = _expect(mode, seeds, offset, horizon)
        acc = {"mae": _round5(fin["overall"]["mae"]),
               "hold": _round5(fin["baselines"]["persistence"]["mae"]),
               "zero": _round5(fin["baselines"]["zero"]["mae"])}
        exact = bool(np.array_equal(acc["mae"], _round5(exp["mae"]))
                     and np.array_equal(acc["hold"], _round5(exp["hold"]))
                     and np.array_equal(acc["zero"], _round5(exp["zero"])))
        if not exact:                     # last-digit float32 drift is allowed, magnitude is not
            exact = bool(np.allclose(acc["mae"], _round5(exp["mae"]), atol=2e-5)
                         and np.allclose(acc["hold"], _round5(exp["hold"]), atol=2e-5)
                         and np.allclose(acc["zero"], _round5(exp["zero"]), atol=2e-5))
        const_exact = None
        if mode == "const":
            const_exact = bool(np.allclose(np.asarray(fin["overall"]["mae"]), abs(offset),
                                           atol=1e-9)
                               and np.allclose(np.asarray(fin["baselines"]["persistence"]["mae"]),
                                               0.0, atol=1e-9))
        buckets_ok = all(np.allclose(_round5(v["mae"]), _round5(exp["bucket_mae"][k]), atol=2e-5)
                         for k, v in fin["by_horizon"].items())
        out[mode] = {
            "accounted_mae": fin["overall"]["mae"],
            "expected_mae_from_fixture_formula": _round5(exp["mae"]).tolist(),
            "accounted_persistence": fin["baselines"]["persistence"]["mae"],
            "expected_persistence": _round5(exp["hold"]).tolist(),
            "accounted_zero": fin["baselines"]["zero"]["mae"],
            "expected_zero": _round5(exp["zero"]).tolist(),
            "bucket_mae_accounted": {k: v["mae"] for k, v in fin["by_horizon"].items()},
            "bucket_mae_expected": {k: v.tolist() for k, v in exp["bucket_mae"].items()},
            "steps_scored": [fin["steps_scored"], exp["scored"]],
            "steps_issued": [fin["steps_issued"], exp["issued"]],
            "contact_steps": [fin["in_contact"]["n"], exp["contact"]],
            "coverage": fin["coverage"],
            "mae_vs_persistence": fin["mae_vs_persistence"],
            "mae_vs_zero": fin["mae_vs_zero"],
            "matches_expected": exact,
        }
        results["checks"].append(("force MAE, persistence and zero baselines match the values "
                                  "computed from the fixture formula (%s world)" % mode, exact))
        results["checks"].append(("horizon-bucket MAEs match per bucket (%s world)" % mode,
                                  bool(buckets_ok)))
        results["checks"].append(("scored/issued/contact step counts match the predicted ones "
                                  "(%s world)" % mode,
                                  [fin["steps_scored"], fin["steps_issued"],
                                   fin["in_contact"]["n"]] == [exp["scored"], exp["issued"],
                                                               exp["contact"]]))
        if mode == "const":
            results["checks"].append(("in the constant-force world the offset predictor's MAE "
                                      "is exactly |offset| and persistence is exactly 0",
                                      bool(const_exact)))
            results["checks"].append(("and every scored step is a contact step, so the contact "
                                      "subset is the whole set",
                                      fin["in_contact"]["n"] == fin["overall"]["n"]))
        else:
            results["checks"].append(("in the ramp world coverage is below 1 because the last "
                                      "chunk outran the episode",
                                      bool(fin["coverage"] is not None and fin["coverage"] < 1.0)))
            results["checks"].append(("the contact subset is a strict subset, so bucketing by "
                                      "contact is not vacuous",
                                      0 < fin["in_contact"]["n"] < fin["overall"]["n"]))
    results["metrics"] = out
    return _run_world("ramp", seeds, offset, horizon)[1]


def check_success_accounting(results, records):
    from eval40k import wilson_ci

    k = sum(1 for r in records if r["success"])
    n = len(records)
    lo, hi = wilson_ci(k, n)
    exp_k = sum(1 for r in records if r["frames"] - 1 == episode_length(r["seed"])
                and succeeds(r["seed"]))
    p = 0.5
    z = 1.96
    hand = ((p + z * z / (2 * 400) - z * math.sqrt(0.25 / 400 + z * z / (4 * 400 * 400)))
            / (1 + z * z / 400))
    corners = [wilson_ci(0, 8), wilson_ci(8, 8), wilson_ci(4, 4)]
    results["success"] = {
        "n": n, "k_accounted": k, "k_from_seed_rule": exp_k,
        "rate": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "wilson_for_p_equals_half_at_n_400_hand": round(hand, 4),
        "wilson_for_p_equals_half_at_n_400_from_code": round(wilson_ci(200, 400)[0], 4),
        "corners": {"k=0,n=8": [round(v, 4) for v in corners[0]],
                    "k=n,n=8": [round(v, 4) for v in corners[1]]},
    }
    results["checks"].append(("successes counted by the harness = successes the fixture's "
                              "seed rule says", k == exp_k))
    results["checks"].append(("Wilson covers the point estimate and stays inside [0,1]",
                              bool(lo <= k / n <= hi and 0.0 <= lo and hi <= 1.0)))
    results["checks"].append(("Wilson CI for a known case matches the closed form",
                              abs(wilson_ci(200, 400)[0] - hand) < 1e-6))
    results["checks"].append(("the k=0 corner gives a strictly positive-free upper bound and "
                              "k=n a bounded lower", bool(corners[0][1] < 1.0 and corners[1][0] > 0.0)))
    eps = sum(1 for r in records if r["nan_abort_frame"] >= 0)
    lengths = sorted({r["frames"] - 1 for r in records})
    results["success"]["episode_lengths"] = lengths
    results["success"]["nan_aborts"] = eps
    results["checks"].append(("no NaN abort in the fixture run (abort detection stays armed "
                              "for the real env)", eps == 0))


def check_contract_gates(results):
    good = {"version": "force-calib-v1", "channels": list(CHANNELS),
            "mean": [0.0] * 6, "std": [1.0] * 6, "scale": 1.0}
    tampered = {
        "wrong version": {**good, "version": "force-calib-v0"},
        "wrong channels": {**good, "channels": [0, 1, 2, 3, 4, 5]},
        "std not positive": {**good, "std": [1, 1, 1, 1, 0, 1]},
        "wrong scale": {**good, "scale": 0.001},
        "std wrong length": {**good, "std": [1.0] * 5},
        "not a dict": ["mean", "std"],
    }
    caught = {name: rejections(check_calib, doc, where=name) for name, doc in tampered.items()}
    accepted = check_calib(good, where="synthetic-good")
    node_bad = rejections(check_node_calibration, {"hist_len": 8, "force_dim": 6, "scale": 1.0},
                          where="node")
    node_ok = check_node_calibration({"hist_len": HIST_LEN, "force_dim": 6, "scale": 1.0},
                                     where="node")
    width = rejections(select_channels, np.zeros(6), where="narrow obs")
    nan = rejections(select_channels, np.full(18, np.nan), where="nan obs")
    results["gates"] = {
        "tampered_calibration_rejected": {k: (v is not None) for k, v in caught.items()},
        "good_calibration_facts": accepted,
        "node_fingerprint_with_hist_len_8_rejected": node_bad,
        "node_fingerprint_accepted": node_ok,
        "six_wide_force_vector_rejected": width,
        "nan_force_vector_rejected": nan,
    }
    results["checks"].append(("every tampered calibration file is refused",
                              all(v is not None for v in caught.values())))
    results["checks"].append(("a wrong hist_len fingerprint is refused", node_bad is not None))
    results["checks"].append(("a too-narrow or NaN force vector from the env is refused",
                              width is not None and nan is not None))
    results["checks"].append(("a well-formed calibration is accepted unchanged",
                              accepted["channels"] == list(CHANNELS)))


def check_noise_seeds(results):
    """Every replan of an episode must draw *fresh* diffusion noise.

    `infer_action` does not keep a sampler state between calls: it builds
    `torch.Generator(...).manual_seed(seed)` afresh (jointwam.py:183-184 for the video and action
    latents, fastwam.py:935 for the force latents) and draws latents of a shape that is constant
    across the episode. So the harness passing its episode seed unchanged means chunk 1..60 all
    start from bit-identical noise -- one unlucky realisation repeated for the whole episode, and
    a distribution the model never saw in training (`torch.randn_like` per window).

    The whole screened table is checked rather than a sample of it, and it is dumped in full: it is
    the reference the eight nodes get compared against afterwards, so a rerun or a node restarted
    mid-block has to reproduce it entry for entry.
    """
    n_chunks = 60                                  # max_steps 600 / replan 10
    screened = list(range(11, 31))                  # the block seed_screen.py certified
    nodes = [5000 * (k + 1) for k in range(8)]      # the eight checkpoint steps of this run
    seqs = {s: [noise_seed(s, c) for c in range(1, n_chunks + 1)] for s in screened}
    table = {f"{s}:{c}": noise_seed(s, c) for s in screened for c in range(1, n_chunks + 1)}
    node_tables = [{f"{s}:{c}": noise_seed(s, c)
                    for s in screened for c in range(1, n_chunks + 1)} for _node in nodes]
    drawn = sorted(table.values())
    collisions = len(drawn) - len(set(drawn))
    in_range = all(isinstance(v, int) and 0 <= v < 2 ** 31 for seq in seqs.values() for v in seq)
    distinct = all(len(set(seq)) == n_chunks for seq in seqs.values())
    stable = all(seqs[s] == [noise_seed(s, c) for c in range(n_chunks, 0, -1)][::-1] for s in screened)
    episodes_differ = all(set(seqs[a]) != set(seqs[b])
                          for i, a in enumerate(screened) for b in screened[i + 1:])
    constant = len(set(noise_seed(11, c) for c in [1] * n_chunks)) == 1
    results["noise"] = {
        "chunks_per_episode": n_chunks,
        "screened_seeds": screened,
        "checkpoint_nodes": nodes,
        "distinct_seeds_for_seed_11": len(set(seqs[11])),
        "first_5_for_seed_11": seqs[11][:5],
        "first_5_for_seed_12": seqs[12][:5],
        "total_draws": n_chunks * len(screened),
        "distinct_draws": len(set(drawn)),
        "collisions_over_full_range": collisions,
        "table_sha256": hashlib.sha256(json.dumps(table, sort_keys=True).encode()).hexdigest(),
        "table": table,
    }
    results["checks"].append(("a chunk seed is a pure function of (episode seed, chunk index), so "
                              "a rerun or a restarted node replays the same noise sequence",
                              stable and constant))
    results["checks"].append(("the replans of one episode never share a chunk seed", distinct))
    results["checks"].append(("every chunk seed is accepted by `manual_seed` (int, 0 <= s < 2**31)",
                              in_range))
    results["checks"].append(("no two screened episodes get the same chunk-seed set",
                              episodes_differ))
    results["checks"].append(("no chunk seed repeats anywhere in the full 20 seed x 60 chunk range",
                              collisions == 0))
    results["checks"].append(("the table takes no checkpoint-node argument, so all eight nodes "
                              "replay one identical episode-seed/chunk-noise table",
                              all(t == table for t in node_tables)))


def check_rollout_seeds(results, seed=11, max_steps=600, replan=10):
    """Watch the real loop hand the seed to `infer_action`, instead of trusting the function.

    `check_noise_seeds` proves `noise_seed` is a good table; it does not prove the harness looks at
    it. So this drives `rollout40k.run_episode` -- the same function the batch evaluation calls --
    with a stand-in model whose only behaviour is recording the `seed=` keyword it receives. Every
    assertion is on that recording, which is what `torch.Generator.manual_seed` would have been
    given. Nothing here touches a GPU, a checkpoint or the simulator: the model is three lines of
    zeros, and the env is a counter.

    The regression is replayed as well: `policy40k.noise_seed` is temporarily replaced by a
    constant, the same episode is rolled again, and the check requires the recording to collapse to
    one seed. A test that only ever sees the fixed code passes whether or not it is armed.
    """
    import torch

    import policy40k

    class Cfg:
        """The two OmegaConf access patterns `ForcePolicy` uses, over a plain dict."""

        def __init__(self, mapping):
            self._mapping = mapping

        def get(self, key, default=None):
            value = self._mapping.get(key, default)
            return Cfg(value) if isinstance(value, dict) else value

        def __getattr__(self, key):
            value = self._mapping[key]
            return Cfg(value) if isinstance(value, dict) else value

    class Bridge:
        DEFAULT_PROMPT = "{task}"

        def prepare_fasterwam_images_official(self, obs):
            return obs["images"], np.zeros((2, 3, 4, 4), dtype=np.float32)

        def build_proprio(self, obs):
            return np.asarray(obs["observation.state"], dtype=np.float32)

        def normalize_proprio(self, proprio, processor):
            return torch.from_numpy(proprio)

        def denormalize_action(self, action, processor):
            return np.asarray(action, dtype=np.float64)

        def quat_wxyz_to_matrix(self, quat):
            return np.eye(3)

    class Adapter:
        def convert(self, action, rot):
            return np.asarray(action, dtype=np.float64)

    class Model:
        torch_dtype = torch.float32

        def __init__(self):
            self.seeds = []

        def infer_action(self, **kwargs):
            self.seeds.append(int(kwargs["seed"]))
            h = int(kwargs["action_horizon"])
            return {"action": np.zeros((1, h, 7), dtype=np.float32),
                    "force_future_newton": np.zeros((1, h, len(CHANNELS)), dtype=np.float32)}

    class Env:
        def __init__(self):
            self.frame = 0

        def step(self, action):
            self.frame += 1
            f = self.frame
            return ({FORCE_OBS_KEY: force18(f), "observation.state": np.zeros(8),
                     "observation.eef_pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
                     "images": []},
                    0.0, f >= max_steps, False, {"success": True, "cookies_in_target": 1})

    def new_policy(model, proprio_fn=None):
        pol = object.__new__(ForcePolicy)
        pol.model = model
        pol.processor = None
        pol.cfg = Cfg({"data": {"train": {"num_frames": 33,
                                          "action_video_freq_ratio": 4}},
                       "EVALUATION": {"rand_device": "cpu", "tiled": False},
                       "eval_num_inference_steps": 10})
        pol.adapter = Adapter()
        pol.device = "cpu"
        pol.bridge = Bridge()
        pol.proprio_fn = proprio_fn
        pol.instruction = "put the cookies in the box"
        pol.gripper_mode = "libero01"
        pol.num_inference_steps = None
        pol.name = "seedprobe"
        pol.stream = ForceStream(hist_len=HIST_LEN)
        pol.calls = 0
        pol.action_horizon = int(pol.cfg.data.train.num_frames) - 1
        pol.num_video_frames = (int(pol.cfg.data.train.num_frames) - 1) \
            // int(pol.cfg.data.train.action_video_freq_ratio) + 1
        return pol

    def reset(env, s):
        return {FORCE_OBS_KEY: force18(0), "observation.state": np.zeros(8),
                "observation.eef_pose": np.array([0., 0., 0., 1., 0., 0., 0.]),
                "images": []}

    def roll(model):
        env = Env()
        record, _measured, chunks = run_episode(env, new_policy(model), seed, reset,
                                                max_steps=max_steps, replan=replan)
        return record, chunks

    model = Model()
    record, chunks = roll(model)
    observed = model.seeds
    expected = [noise_seed(seed, c) for c in range(1, len(observed) + 1)]
    n_chunks = len(observed)
    chunk_index_ok = (n_chunks == max_steps // replan
                      and [c["meta"]["replan_from_frame"] for c in chunks]
                      == [i * replan for i in range(n_chunks)])
    seeds_advance = observed == expected and len(set(observed)) == n_chunks
    meta_matches_calls = all(chunks[i]["meta"]["noise_seed"] == observed[i]
                             for i in range(n_chunks)) and n_chunks == record["decision_chunks"]

    frozen = policy40k.noise_seed
    try:
        policy40k.noise_seed = lambda s, c: 7
        replayed = Model()
        roll(replayed)
    finally:
        policy40k.noise_seed = frozen
    caught = len(set(replayed.seeds)) == 1 and len(set(observed)) > 1

    results["rollout_noise"] = {
        "seed": int(seed),
        "max_steps": int(max_steps),
        "replan": int(replan),
        "decision_chunks": int(record["decision_chunks"]),
        "frames_observed": int(record["frames"]),
        "infer_action_calls": int(n_chunks),
        "first_5_seeds_seen_by_infer_action": observed[:5],
        "distinct_sees_seen_by_infer_action": len(set(observed)),
        "replan_frames": [c["meta"]["replan_from_frame"] for c in chunks][:5],
        "regression_replay_distinct_seeds": len(set(replayed.seeds)),
    }
    results["checks"].append(("the real loop requests one chunk per replan and the frame it is "
                              "issued from advances by exactly `replan`", chunk_index_ok))
    results["checks"].append(("`infer_action` receives `noise_seed(episode seed, chunk index)` on "
                              "every call, and `meta.noise_seed` is the same value",
                              seeds_advance and meta_matches_calls))
    results["checks"].append(("replaying the old behaviour -- one seed for the whole episode -- is "
                              "caught by this test rather than passing it", caught))

    # the proprio source: `proprio_fn` is how the real drivers hand the model the 14 training
    # columns, so the routing has to be tested here, in the one CPU test that runs without weights
    wide = np.arange(14, dtype=np.float32)
    used = []

    def sixteen_wide(_obs):
        used.append(1)
        return wide

    obs0 = reset(None, 0)
    hist0 = np.zeros((HIST_LEN, len(CHANNELS)), dtype=np.float32)
    mask0 = np.ones(HIST_LEN, dtype=bool)
    overridden = new_policy(Model(), proprio_fn=sixteen_wide)._infer_kwargs(obs0, hist0, mask0, 3)
    default_path = new_policy(Model())._infer_kwargs(obs0, hist0, mask0, 3)
    results["proprio_routing"] = {
        "with_proprio_fn_shape": list(np.asarray(overridden["proprio"]).shape),
        "with_proprio_fn_calls": len(used),
        "without_proprio_fn_shape": list(np.asarray(default_path["proprio"]).shape)}
    results["checks"].append(("a driver that passes `proprio_fn` has its proprio handed to "
                              "`infer_action` instead of the bridge's own builder -- this is the "
                              "call site the batch evaluator uses for the 14-wide training state",
                              len(used) == 1
                              and np.asarray(overridden["proprio"]).reshape(-1).shape == (14,)
                              and np.array_equal(np.asarray(overridden["proprio"]).reshape(-1),
                                                 wide)))
    results["checks"].append(("with no override the policy still asks the bridge, so the CPU "
                              "fixtures and any 8-wide deployment keep working",
                              np.asarray(default_path["proprio"]).reshape(-1).shape == (8,)))


# --- text conditioning ---------------------------------------------------------
#
# The first real smoke run died inside `infer_action` because the model was built from
# `run40k/config.yaml`, which sets `model.load_text_encoder: false` -- training reads text
# conditioning from the precomputed embedding cache and never wants the T5 in memory, but
# `infer_a3_worker` passes a `prompt`, and `encode_prompt` raises when no encoder is loaded.
# Fixing that raises a second question the harness must not answer by guessing: *which* prompt.
# These checks cover both halves -- where the inference knob comes from, and what the instruction
# is -- against synthetic corpora built in a temp directory, plus the real one when it is present.

TEXT_TEMPLATE = "a robot executes the following instruction: {task}"
TEXT_TASK = "transfer 10 cookies into target box"
TEXT_A3_TARGET = "fasterwam.datasets.lerobot.a3_video_dataset.A3VideoDataset"
# The one training prompt measured on the pod 2026-11-12: the only embedding file in
# /workspace/a3_data/text_embeds_cache. A tripwire, not an input -- if the corpus or the cache
# directory changes underneath the harness, this check is the thing that says so.
TEXT_REAL_SHA256 = "b6f1657eeb7e8f5a4e4310074fab14ad8d79ed3ef798f677be2c137817d0f7b6"


class TextConf(dict):
    """A config that answers `.key` and `.get(...)` like the OmegaConf node `build_policy` passes.

    Nesting is wrapped on the way in because the code under test walks the tree by attribute
    (`doc.data.train._target_`), and attribute assignment is kept in step with item assignment
    because `deployment_text_encoder_settings` writes `doc.model.load_text_encoder = True` and the
    record reads it back through `.get`. This stands in for a `DictConfig` only where `omegaconf`
    is genuinely absent; see `conf_node`.
    """

    def __init__(self, mapping=None):
        super().__init__()
        for key, value in dict(mapping or {}).items():
            self[key] = value

    def __setitem__(self, key, value):
        super().__setitem__(key, _wrap_conf(value))

    def __setattr__(self, name, value):
        self[name] = value

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _wrap_conf(value):
    if isinstance(value, dict) and not isinstance(value, TextConf):
        return TextConf(value)
    if isinstance(value, (list, tuple)):
        return [_wrap_conf(v) for v in value]
    return value


def conf_node(mapping):
    """Build the synthetic config as the object the function under test actually expects.

    `deployment_text_encoder_settings` locks and unlocks the struct flag through
    `OmegaConf.set_struct`, which only a real config node answers; handing it an attribute-getting
    dict raised `AttributeError: _set_flag` on the first pod run. So where the real `omegaconf` is
    installed the node is a genuine `DictConfig` and every assertion below runs against the same
    type the drivers build from `run40k/config.yaml`. Only on the machine that has no `omegaconf`
    at all -- where `check_text_conditioning` installs a no-op stub for it -- does the plain
    `TextConf` stand in, and the stub is marked so this function can tell the two apart.
    """
    try:
        import omegaconf as _oc

        if not getattr(_oc, "QODER_FLOW_TEST_STUB", False) and hasattr(_oc.OmegaConf, "create"):
            return _oc.OmegaConf.create(mapping)
    except Exception:
        pass
    return TextConf(mapping)


def text_corpus(root, tasks=None, cache_names=(), context_len=128, n_rows=24,
                target=TEXT_A3_TARGET, n_dirs=1, write_arrays=True):
    """A stand-in training corpus: `arrays.npz` plus the embedding cache the loader would read.

    Returns `(doc, arrays_path, cache_dir)`. `doc` carries exactly the four keys the prompt code
    reads -- `data.train._target_`, `dataset_dirs`, `context_len`, `text_embedding_cache_dir` --
    and a `model.load_text_encoder: false`, which is what the run config really says.
    """
    root = Path(root)
    data_dir = root / "v1g"
    cache_dir = root / "cache"
    data_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays = data_dir / "arrays.npz"
    if write_arrays:
        payload = {"action": np.zeros((n_rows, 7), dtype=np.float32)}
        if tasks is not None:
            payload["task_str"] = np.array(tasks)
        np.savez(str(arrays), **payload)
    for name in cache_names:
        (cache_dir / name).write_bytes(b"placeholder-not-an-embedding")
    dirs = [str(data_dir)] * n_dirs
    doc = conf_node({"model": {"load_text_encoder": False,
                               "model_id": "Wan-AI/Wan2.2-TI2V-5B",
                               "tokenizer_model_id": "Wan-AI/wan2.2-ti2v-5b-tokenizer",
                               "tokenizer_max_len": 128,
                               "redirect_common_files": False,
                               "skip_dit_load_from_pretrain": True,
                               "action_dit_pretrained_path": None},
                     "data": {"train": {
                         "_target_": target,
                         "dataset_dirs": dirs,
                         "context_len": context_len,
                         "text_embedding_cache_dir": str(cache_dir)}}})
    return doc, arrays, cache_dir


def text_cache_name(task, context_len=128, template=TEXT_TEMPLATE):
    return "%s.t5_len%d.wan22ti2v5b.pt" % (
        hashlib.sha256(template.format(task=task).encode("utf-8")).hexdigest(), context_len)


def system_exit_message(fn, *args, **kw):
    """Return the message if `fn` refuses, else None -- these gates are required to refuse."""
    try:
        fn(*args, **kw)
    except SystemExit as exc:
        return str(exc)
    return None


def source_before(text, first, second):
    """True when both markers are present and `first` comes first; a missing marker is False."""
    try:
        return text.index(first) < text.index(second)
    except ValueError:
        return False


def source_string_constant(path, name):
    """Read a module-level `NAME = "..."` out of a source file without importing it."""
    import ast

    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError("%s defines no module-level string constant %s" % (path, name))


def check_text_conditioning(results):
    """Where the inference-only encoder key comes from, and which prompt the weights know.

    Nothing here needs a GPU, a checkpoint or the simulator, and nothing here imports the model
    package: the two constants the real corpus is interpreted with are read out of the dataset
    source with `ast`, which is the same text the trainer imported. The bit-exactness half uses a
    small synthetic cache and a stand-in encoder, because the property being tested is the
    comparison, not the tensor size; the real encoder against the real cache is the job of
    `verify_prompt_conditioning` inside `build_policy`, which the smoke run executes.
    """
    import tempfile

    import eval40k as E

    add = lambda desc, ok: results["checks"].append((desc, ok))  # noqa: E731
    out = {"synthetic": {}, "real_corpus": {}, "torch": "absent", "omegaconf": "absent"}

    # The real package is not importable on the stub-torch pre-check machine, and the two
    # functions below import torch / omegaconf inside their bodies. Substitute only what the
    # checked lines touch, say so in the record, and let the pod run be the acceptance gate.
    try:
        import torch as _probe
        have_real_torch = hasattr(_probe, "load") and hasattr(_probe, "bool")
    except Exception:
        have_real_torch = False
    out["torch"] = "real" if have_real_torch else "stub or absent"
    try:
        import omegaconf as _oc
        have_omegaconf = hasattr(_oc, "OmegaConf") and hasattr(_oc.OmegaConf, "load")
    except Exception:
        have_omegaconf = False
    stubbed_omegaconf = not have_omegaconf
    if stubbed_omegaconf:
        import types

        mod = types.ModuleType("omegaconf")

        class _StubOmega:
            @staticmethod
            def set_struct(_cfg, _value):
                return None

            @staticmethod
            def to_container(cfg, resolve=False):
                return json.loads(json.dumps(cfg))

            @staticmethod
            def load(_path):
                raise NotImplementedError("flow_test does not read a real config without omegaconf")

        mod.OmegaConf = _StubOmega
        mod.QODER_FLOW_TEST_STUB = True
        sys.modules["omegaconf"] = mod
    out["omegaconf"] = "stubbed (no omegaconf here)" if stubbed_omegaconf else "real"

    with tempfile.TemporaryDirectory() as tmp:
        # 1. the file-name recipe, against an independent re-derivation
        prompt, sha = E.hashed_prompt(TEXT_TASK, TEXT_TEMPLATE)
        name = E.prompt_cache_name(sha, 128)
        own = TEXT_TEMPLATE.format(task=TEXT_TASK)
        own_sha = hashlib.sha256(own.encode("utf-8")).hexdigest()
        out["recipe"] = {"prompt": prompt, "sha256": sha, "cache_file": name}
        add("the cached embedding is looked up by the hash of the *formatted* prompt, the way "
            "`robot_video_dataset.py:251-252` opens it, and the file name carries the context "
            "length and the encoder tag",
            prompt == own and sha == own_sha
            and name == "%s.t5_len128.wan22ti2v5b.pt" % own_sha
            and sha != hashlib.sha256(TEXT_TASK.encode("utf-8")).hexdigest())
        add("a different instruction hashes to a different embedding file, so conditioning on one "
            "the weights never saw cannot silently read the training cache",
            E.hashed_prompt("pick up the red block", TEXT_TEMPLATE)[1] != sha)

        # 2. the instruction is read off the corpus, both ways it can be stored
        good_name = text_cache_name(TEXT_TASK)
        doc, arrays, cache = text_corpus(Path(tmp) / "no_task_str", cache_names=[good_name])
        out["synthetic_doc_class"] = type(doc).__name__
        add("the synthetic corpus is handed over as the config type the drivers really load -- a "
            "genuine `DictConfig` wherever `omegaconf` is installed, and the attribute-getting "
            "stand-in only under this file's own stub, which is the case where `set_struct` is a "
            "no-op",
            out["synthetic_doc_class"] == ("TextConf" if stubbed_omegaconf else "DictConfig"))
        facts = E.training_prompt_facts(doc, TEXT_TASK, TEXT_TEMPLATE)
        out["synthetic"]["no_task_str"] = facts
        add("a corpus with no `task_str` falls back to the dataset module's default task, and the "
            "fallback is recorded as measured rather than assumed -- the array was opened",
            facts["task"] == TEXT_TASK and "no task_str" in facts["task_source"]
            and facts["prompt_sha256"] == sha and facts["cache_file"] == good_name
            and facts["cache_files_present"] == [good_name] and facts["rows"] == 24
            and facts["context_len"] == 128 and facts["dataset_target"] == TEXT_A3_TARGET)
        add("the arrays the config names are the arrays that were read, not a path the harness "
            "guessed", facts["arrays"] == str(arrays) and facts["cache_dir"] == str(cache))

        other = "stack the cups by size"
        doc2, arrays2, _c2 = text_corpus(Path(tmp) / "with_task_str",
                                         tasks=[other] * 24,
                                         cache_names=[text_cache_name(other)])
        facts2 = E.training_prompt_facts(doc2, TEXT_TASK, TEXT_TEMPLATE)
        out["synthetic"]["with_task_str"] = facts2
        add("when the corpus does carry `task_str` the data wins over the module default, so the "
            "instruction cannot be set by whichever copy of the package happened to be imported",
            facts2["task"] == other and facts2["task_source"] == "arrays.npz:task_str"
            and facts2["prompt_sha256"] != sha)

        # 3. every way the corpus can fail to describe one prompt has to stop the run
        refusals = {
            "two distinct task strings": (
                text_corpus(Path(tmp) / "r_multi",
                            tasks=[TEXT_TASK, other] * 12, cache_names=[good_name]), TEXT_TASK),
            "no cached embedding for the training prompt": (
                text_corpus(Path(tmp) / "r_nocache", tasks=[other] * 24,
                            cache_names=[good_name]), TEXT_TASK),
            "two embedding files in the cache directory": (
                text_corpus(Path(tmp) / "r_twocache", cache_names=[good_name, "z" * 64
                                                                   + ".t5_len128.wan22ti2v5b.pt"]),
                TEXT_TASK),
            "a dataset class other than the one whose rule is implemented": (
                text_corpus(Path(tmp) / "r_target", cache_names=[good_name],
                            target="fasterwam.datasets.lerobot.RobotVideoDataset"), TEXT_TASK),
            "two dataset directories": (
                text_corpus(Path(tmp) / "r_dirs", cache_names=[good_name], n_dirs=2), TEXT_TASK),
            "arrays.npz missing": (
                text_corpus(Path(tmp) / "r_arrays", cache_names=[good_name],
                            write_arrays=False), TEXT_TASK),
        }
        markers = {"two distinct task strings": "distinct task strings",
                   "no cached embedding for the training prompt": "no cached training embedding",
                   "two embedding files in the cache directory": "holds 2 embedding files",
                   "a dataset class other than the one whose rule is implemented":
                       "not the A3 dataset",
                   "two dataset directories": "exactly one dataset_dirs entry",
                   "arrays.npz missing": "is not readable"}
        got = {}
        for label, ((doc_r, arrays_r, _cache_r), default_task) in refusals.items():
            msg = system_exit_message(E.training_prompt_facts, doc_r, default_task, TEXT_TEMPLATE)
            got[label] = msg
            add("the harness refuses %s instead of picking an instruction" % label,
                msg is not None and markers[label] in msg)
        out["refusals"] = {k: (v.split("\n")[0][:120] if v else None) for k, v in got.items()}

        # 4. `--instruction` is not a free knob
        add("`--instruction` left at its default resolves to the corpus instruction",
            E.check_instruction(None, facts) == TEXT_TASK)
        add("an `--instruction` that repeats the corpus instruction is accepted and normalised to "
            "the same string", E.check_instruction(TEXT_TASK, facts) == TEXT_TASK)
        msg = system_exit_message(E.check_instruction, other, facts)
        out["invented_instruction_rejected"] = msg
        add("an `--instruction` that is not the training instruction is refused, naming the "
            "training instruction and the cache directory it has no embedding in",
            msg is not None and "is not the instruction these weights were trained on" in msg
            and TEXT_TASK in msg and str(cache) in msg)

        # 5. the inference-only encoder key, taken from the deployment config and nowhere else
        dep = Path(tmp) / "sim_libero.yaml"
        dep.write_text("model:\n"
                      "  load_text_encoder: true\n"
                      "  model_id: Some/OtherModel\n"
                      "  skip_dit_load_from_pretrain: false\n"
                      "  action_dit_pretrained_path: /some/other.pt\n", encoding="utf-8")
        tf = E.deployment_text_encoder_settings(doc, deploy_cfg=dep)
        out["encoder_settings"] = tf
        add("the only key taken from the deployment config is `model.load_text_encoder`, applied on "
            "top of this run's config -- the run config's `model_id` and DiT weight keys are left "
            "as they are, because they decide which weights the architecture is built around",
            tf["train_config_value"] is False and tf["deployment_value"] is True
            and tf["applied"] is True and doc.model.load_text_encoder is True
            and tf["keys_left_on_run_config"]["model_id"] == "Wan-AI/Wan2.2-TI2V-5B"
            and tf["keys_left_on_run_config"]["skip_dit_load_from_pretrain"] is True
            and tf["keys_left_on_run_config"]["action_dit_pretrained_path"] is None)
        dep.write_text("model:\n  load_text_encoder: false\n", encoding="utf-8")
        msg = system_exit_message(E.deployment_text_encoder_settings, doc, dep)
        add("a deployment config that also leaves the encoder unloaded is refused rather than "
            "silently producing a model that cannot take a prompt",
            msg is not None and "load_text_encoder=false" in msg)
        dep.write_text("model:\n  model_id: Some/OtherModel\n", encoding="utf-8")
        msg = system_exit_message(E.deployment_text_encoder_settings, doc, dep)
        add("a deployment config with no `load_text_encoder` key at all is refused: the harness "
            "has no other source for it and will not invent one",
            msg is not None and "no model.load_text_encoder" in msg)

        # 6. the bit-exactness gate itself, against a stand-in encoder
        if have_real_torch:
            import torch

            ctx_len, dim, n_real = 8, 6, 5
            junk = torch.arange(ctx_len * dim, dtype=torch.float32).reshape(ctx_len, dim)
            cached = junk.clone()
            cached[:n_real] += 1.0               # real rows differ from the padding rows
            cmask = torch.zeros(ctx_len, dtype=torch.bool)
            cmask[:n_real] = True
            training_context = cached.clone()
            training_context[~cmask] = 0.0

            class Encoder:
                def __init__(self, emb, mask=None):
                    self.emb, self.mask = emb, mask

                def encode_prompt(self, _prompt):
                    return self.emb, (torch.ones(1, ctx_len, dtype=torch.bool)
                                      if self.mask is None else self.mask)

            cache_file = Path(tmp) / "tiny.pt"
            torch.save({"context": cached, "mask": cmask}, str(cache_file))
            tiny = {"cache_dir": tmp, "cache_file": cache_file.name, "prompt": "p"}
            ok = E.verify_prompt_conditioning(Encoder(training_context.unsqueeze(0)), tiny)
            out["verify_exact"] = ok
            add("the check compares `encode_prompt` against the cache *after* the dataset's own "
                "`context[~mask] = 0` transform, so the padding the loader zeroed at training time "
                "is zeroed here too and matching bits really mean matching conditioning",
                ok["n_diff_elements"] == 0 and ok["max_abs_diff"] == 0.0
                and ok["cached_mask_true"] == n_real == ok["encoded_nonzero_token_rows"]
                and ok["encoded_mask_all_true"] is True
                and ok["cached_shape"] == [ctx_len, dim]
                and float(cached[n_real:].abs().max()) > 0.0)
            flipped = training_context.clone()
            flipped[0, 0] += 1e-3
            msg = system_exit_message(E.verify_prompt_conditioning,
                                      Encoder(flipped.unsqueeze(0)), tiny)
            add("one differing element is refused: the encoder, the tokenizer length or the device "
                "is not what wrote the cache, and evaluating through it would change the text "
                "conditioning rather than measure the policy under it",
                msg is not None and "does not reproduce the cached training embedding" in msg)
            msg = system_exit_message(E.verify_prompt_conditioning,
                                      Encoder(torch.zeros(1, ctx_len // 2, dim)), tiny)
            add("a length-encoder output that does not match the cache is refused as not "
                "comparable rather than partially compared",
                msg is not None and "not comparable" in msg)
            nan_cache = cached.clone()
            nan_cache[1, 1] = float("nan")
            nan_file = Path(tmp) / "nan.pt"
            torch.save({"context": nan_cache, "mask": cmask}, str(nan_file))
            msg = system_exit_message(E.verify_prompt_conditioning,
                                      Encoder(torch.zeros(1, ctx_len, dim)),
                                      {"cache_dir": tmp, "cache_file": nan_file.name,
                                       "prompt": "p"})
            add("NaN anywhere in the cached conditioning aborts before any rollout is scored",
                msg is not None and "NaN in the text conditioning" in msg)
            zr_cached = cached.clone()
            zr_cached[2] = 0.0                  # a real token whose embedding is all zeros
            zr_file = Path(tmp) / "zero_row.pt"
            torch.save({"context": zr_cached, "mask": cmask}, str(zr_file))
            zr_context = zr_cached.clone()
            zr_context[~cmask] = 0.0
            msg = system_exit_message(E.verify_prompt_conditioning,
                                      Encoder(zr_context.unsqueeze(0)),
                                      {"cache_dir": tmp, "cache_file": zr_file.name,
                                       "prompt": "p"})
            add("even with every bit matching, an encoder that lands a different number of real "
                "tokens than the training mask counted is refused -- that is a different prompt",
                msg is not None and "non-zero token rows" in msg)
        else:
            print("NOTE  the bit-exactness block needs a real torch (this machine has %s); the "
                  "pod run of flow_test.py is the one that registers those 5 checks, and "
                  "`verify_prompt_conditioning` also runs for real inside `build_policy`"
                  % out["torch"], flush=True)
    if stubbed_omegaconf:
        sys.modules.pop("omegaconf", None)

    # 7. the same reading applied to the corpus that actually trained these weights
    real_cfg, real_arrays = Path(E.DEFAULT_CFG), Path(E.V1G_ARRAYS)
    ds_dir = Path(E.FW) / "src" / "fasterwam" / "datasets" / "lerobot"
    if real_arrays.is_file() and real_cfg.is_file() and not stubbed_omegaconf \
            and (ds_dir / "a3_video_dataset.py").is_file():
        from omegaconf import OmegaConf

        task_real = source_string_constant(ds_dir / "a3_video_dataset.py", "DEFAULT_A3_TASK")
        tmpl_real = source_string_constant(ds_dir / "robot_video_dataset.py", "DEFAULT_PROMPT")
        rfacts = E.training_prompt_facts(OmegaConf.load(str(real_cfg)), task_real, tmpl_real)
        out["real_corpus"] = {k: rfacts[k] for k in
                              ("task", "task_source", "prompt_sha256", "context_len", "cache_dir",
                               "cache_file", "cache_files_present", "rows", "arrays")}
        add("on the pod: the corpus that trained these weights carries no `task_str`, so every "
            "window was conditioned on the dataset module's default task and exactly one prompt is "
            "in play",
            rfacts["task"] == task_real and "no task_str" in rfacts["task_source"]
            and rfacts["rows"] == 42681)
        add("on the pod: the prompt the harness would build hashes to the single embedding file in "
            "the training cache directory -- the file name is the evidence, and it is the one "
            "measured on 2026-11-12",
            rfacts["prompt_sha256"] == TEXT_REAL_SHA256
            and rfacts["cache_files_present"] == [rfacts["cache_file"]]
            and rfacts["cache_file"] == "%s.t5_len%d.wan22ti2v5b.pt" % (TEXT_REAL_SHA256, 128)
            and rfacts["context_len"] == 128)
        add("on the pod: the arrays read for the task string are the ones the training config "
            "names, and the dataset class named there is the class whose task rule was applied",
            Path(rfacts["arrays"]).resolve() == real_arrays.resolve()
            and rfacts["dataset_target"].endswith("a3_video_dataset.A3VideoDataset"))
    else:
        print("NOTE  the real-corpus block needs this pod's %s and %s (present: %s, %s); it is the "
              "CPU reading of the frozen corpus, and the smoke run re-checks it against the encoder"
              % (real_cfg, real_arrays, real_cfg.is_file(), real_arrays.is_file()), flush=True)

    # 8. the ordering that makes the above the path every real run takes
    src = (HERE / "eval40k.py").read_text(encoding="utf-8")
    bp = src[src.index("def build_policy"):]
    add("the encoder key is taken from the deployment config before the model is instantiated, so "
        "`instantiate` builds a module that can hold a text encoder at all",
        source_before(bp, "deployment_text_encoder_settings(doc)", "instantiate(doc.model"))
    add("the prompt provenance is resolved, and the requested instruction checked against it, "
        "before the 11 GB artifact is opened",
        source_before(bp, "TEXT_PROMPT_FACTS.update(resolve_prompt_facts(doc))",
                      "model.load_checkpoint(str(merged))")
        and source_before(bp, "instruction = check_instruction",
                          "model.load_checkpoint(str(merged))"))
    add("the weight search path is anchored and each component file proved present before the "
        "checkpoint load, so a missing encoder fails without a download and without GPU work",
        source_before(bp, "anchor_weight_search_path(doc)", "model.load_checkpoint(str(merged))"))
    add("the conditioning is verified after the module is on the device and in eval mode, and "
        "before the force normalisation stats are attached -- an unverified prompt never reaches a "
        "rollout",
        source_before(bp, "model = model.to(device).eval()",
                      "verify_prompt_conditioning(model, TEXT_PROMPT_FACTS)")
        and source_before(bp, "verify_prompt_conditioning(model, TEXT_PROMPT_FACTS)",
                          "set_force_norm_stats(calib["))
    add("the policy is handed the resolved instruction, never the raw argument, and the env "
        "backend resolves it too so `--dry-run` reports what the run will actually condition on",
        "instruction=instruction," in bp and "instruction=args.instruction" not in bp
        and "args.instruction = check_instruction(args.instruction, TEXT_PROMPT_FACTS)" in src)
    add("`--instruction` defaults to None in both drivers, so neither can run on a prompt the "
        "corpus did not supply",
        'ap.add_argument("--instruction", default=None,' in src
        and 'ap.add_argument("--instruction", default=None,'
            in (HERE / "smoke40k.py").read_text(encoding="utf-8"))
    batch_src = (HERE / "batch_eval.py").read_text(encoding="utf-8")
    add("the batch driver launches `eval40k.py` without an `--instruction` argument at all, so the "
        "160-episode run cannot drift onto a prompt the smoke test did not verify -- this is the "
        "formal-eval path, not just the smoke tool",
        '"--instruction"' not in batch_src and "eval40k.py" in batch_src)
    sm = (HERE / "smoke40k.py").read_text(encoding="utf-8")
    add("the smoke cross-check compares the harness prompt against the worker's template applied "
        "to the *resolved* instruction, and reports the deviation from the worker's own default "
        "instead of hiding it",
        "cross_prompt(policy, W)" in sm
        and 'W.DEFAULT_PROMPT.format(task=policy.instruction)' in sm
        and '"item": "instruction"' in sm)
    results["text_conditioning"] = out
    return out


def check_episode_reporting(results):
    """The per-episode summary has to survive a real rollout, not only a fixture one.

    Two live smokes were lost at this point in `smoke40k.py` with the GPU work already done: the
    report stage aggregated the chunk log with a set comprehension over the lists the recorder writes
    (`TypeError: unhashable type: 'list'`), and then read a gripper field called `gripper_in_pm1`
    while the recorder writes `gripper_is_pm1_binary` (`KeyError`). Neither is visible from the file
    it lives in, and the second was invisible to the first fix because the shape bug killed the run
    one line earlier. So this drives the *real* `install_recorder`, the *real*
    `install_chunk_recorder` and the *real* `summarize_episode`, and derives the names the report
    consumes from the smoke's own source: a third rename fails here, on any machine, instead of
    there, five minutes and one model load after the last episode was rolled.
    """
    import smoke40k as SM

    horizon, adim, fdim, episode_seed = 32, 14, len(CHANNELS), 11
    sm = (HERE / "smoke40k.py").read_text(encoding="utf-8")
    p4 = (HERE / "policy40k.py").read_text(encoding="utf-8")
    calls, clog, chunks = [], [], []

    class FakeModel:
        """`infer_action` with the real return keys: a (1, H, 7) action and a (1, H, 6) force."""

        class Out(np.ndarray):
            """A model output the recorder can read without torch: it asks `act.requires_grad`."""

            requires_grad = False

        def infer_action(self, **kwargs):
            return {"action": np.zeros((1, kwargs["action_horizon"], 7)).view(FakeModel.Out),
                    "force_future_newton": np.zeros((1, horizon, fdim))}

    class FakePolicy:
        """`chunk` with the return contract `ForcePolicy.chunk` has: env actions, (H, 6) force, meta."""

        def chunk(self, obs, frame, seed):
            n = len(chunks) + 1
            actions = [np.concatenate([np.zeros(adim - 1), [1.0 if i % 2 else -1.0]])
                       for i in range(horizon)]
            meta = {"infer_s": 0.5 * n, "replan_from_frame": int(frame),
                    "noise_seed": int(noise_seed(episode_seed, n)), "action_absmax": 1.0,
                    "force_feedback_applied": False, "history_mask_true": min(n * 4, HIST_LEN)}
            return actions, np.full((horizon, fdim), float(n)), meta

    err, out, rec, before = None, None, None, set()
    try:
        model, policy = FakeModel(), FakePolicy()
        SM.install_recorder(model, calls)
        SM.install_chunk_recorder(policy, clog)
        for i in range(3):
            model.infer_action(seed=noise_seed(episode_seed, i + 1), rand_device="cuda:0",
                               num_inference_steps=16, action_horizon=horizon,
                               num_video_frames=16, input_image=np.zeros((1, 3, 8, 8)),
                               proprio=np.linspace(-1.0, 1.0, adim)[None, :],
                               force_history=np.zeros((1, HIST_LEN, fdim)),
                               force_history_mask=np.arange(HIST_LEN)[None, :] < (i + 1) * 4)
            actions, pred_force, meta = policy.chunk(None, frame=i * 10, seed=episode_seed)
            chunks.append({"frame": i * 10, "pred_force": pred_force,
                           "n_actions": len(actions), "meta": meta})
        rec, _measured, _stub_chunks = run_episode(
            FixtureEnv(seed=episode_seed, frames=41), StubPolicy(pred_mode="zero"),
            seed=episode_seed, reset=lambda e, s: e.reset(seed=s), max_steps=41, replan=10)
        rec["scene_key"] = "fixture-scene"           # what main adds around, not inside, the fold
        rec["force_steps_scored"] = 0
        before = set(rec)
        out = SM.summarize_episode(rec, calls, clog, chunks)
    except Exception as exc:                                     # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)

    tail = sm.split('res["episodes"] = episodes')[-1]
    consumed = (set(re.findall(r'\br\["([a-z0-9_]+)"\](?!=)', tail))
                | set(re.findall(r'\brec\["([a-z0-9_]+)"\](?!=)', tail)))
    missing = sorted(consumed - set(rec or ()))
    summary_keys = sorted(set(out or {}) - before)
    meta_keys = ["noise_seed", "action_absmax", "infer_s", "replan_from_frame"]
    clog_keys = sorted({k for c in clog for k in c})
    results["episode_reporting"] = {
        "error": err, "infer_calls_recorded": len(calls), "chunks_recorded": len(clog),
        "chunk_log_keys": clog_keys, "consumed_after_episodes": sorted(consumed),
        "never_written": missing, "written_by_the_fold": summary_keys,
        "gripper_values": None if out is None else out["gripper_pm1_values"],
        "chunk_shapes": None if out is None else out["chunk_shapes"]}
    results["checks"].append((
        "the real recorders and the real per-episode fold agree field by field: three proxy calls "
        "and three recorded chunks pass through `summarize_episode` without a KeyError or a "
        "TypeError, which is exactly how the last two live smokes died",
        err is None and len(calls) == 3 and len(clog) == 3 and len(chunks) == 3))
    results["checks"].append((
        "every key the report reads after the episodes is written by `run_episode`, by `main` "
        "around the fold, or by the fold itself -- the name list is taken from the smoke's source, "
        "so renaming a producer without renaming the report fails on CPU",
        bool(consumed) and not missing))
    results["checks"].append((
        "the gripper aggregate reads the recorder's own key (not the `gripper_in_pm1` that raised): "
        "it is True exactly when every recorded chunk says so, over the +/-1 column the fixture "
        "actually emits",
        out is not None and out["gripper_is_pm1_binary"] is True
        and all(c["gripper_is_pm1_binary"] for c in clog)
        and out["gripper_pm1_values"] == [-1.0, 1.0]))
    results["checks"].append((
        "the fold ties the seed the model was handed to the seed the policy asked for, both taken "
        "from the producers: %d replans, distinct, matching `noise_seed(episode seed, k)`"
        % len(calls),
        out is not None and out["seeds_seen_by_model"] == out["expected_seeds"]
        == out["meta_noise_seed"] and out["distinct_seeds"] == len(calls)))
    results["checks"].append((
        "the meta keys the fold reads from a chunk are the keys `ForcePolicy.chunk` really returns "
        "(the cross-file half of the same contract): %s" % meta_keys,
        all('"%s":' % k in p4 for k in meta_keys)))
    results["checks"].append((
        "the chunk-shape aggregation takes the lists the recorder really writes and returns them "
        "de-duplicated and sorted, so a whole episode of one shape reports exactly one shape",
        SM.distinct_chunk_shapes(clog) == [[horizon, adim]]
        and SM.distinct_chunk_shapes(clog + [{"chunk_shape": [8, 6]}]) == [[8, 6], [horizon, adim]]))
    results["checks"].append((
        "the episode record goes through that function, and no set comprehension over lists is "
        "left anywhere in the smoke -- the failure mode was only reachable on a real rollout",
        'rec["chunk_shapes"] = distinct_chunk_shapes(clog)' in sm
        and "sorted({list(" not in sm
        and '"chunk_shape": list(arr.shape)' in sm))


def run():
    results = {"checks": [], "python": sys.version.split()[0],
               "numpy": np.__version__}
    measured = check_padding(results)
    check_time_base_guard(results)
    records = check_metrics(results)
    check_success_accounting(results, records)
    check_contract_gates(results)
    check_noise_seeds(results)
    check_rollout_seeds(results)
    check_text_conditioning(results)
    check_episode_reporting(results)
    results["n_checks"] = len(results["checks"])
    results["n_passed"] = sum(1 for _d, ok in results["checks"] if ok)
    results["failed"] = [d for d, ok in results["checks"] if not ok]
    results["all_passed"] = results["n_passed"] == results["n_checks"]
    for desc, ok in results["checks"]:
        print(("PASS  " if ok else "FAIL  ") + desc, flush=True)
    print(json.dumps({"n_checks": results["n_checks"], "n_passed": results["n_passed"],
                      "all_passed": results["all_passed"], "failed": results["failed"]}))
    return results


def main():
    out = None
    for i, a in enumerate(sys.argv):
        if a == "--json" and i + 1 < len(sys.argv):
            out = Path(sys.argv[i + 1])
    results = run()
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, sort_keys=True, default=str))
        print(json.dumps({"json": str(out)}))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
