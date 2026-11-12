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
* the three things the first live rollout of a real checkpoint blamed on the model: the shape it
  returns (`[T, D]`, proved against the model's own source and driven through the real recorders in
  both orientations), the device it sits on (`cuda` and `cuda:0` are one card), the scene it started
  from (hashed at reset, before the robot moves) and the wall-clock split the batch is sized from.

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

# The batch driver is importable without torch: it only reads the volume and the node state. That is
# what lets the disk gate it refuses to start without be tested here instead of at 03:00 on a GPU.
import batch_eval as BE  # noqa: E402
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
        """`infer_action` with the real return contract: a `[H, 7]` action and a `[H, 6]` force.

        Both are batch-dim-free because the model indexes `latents[0]` before returning it -- see
        `check_inference_contract`, which reads that out of the model's source.
        """

        class Out(np.ndarray):
            """A model output the recorder can read without torch: it asks `act.requires_grad`."""

            requires_grad = False

        def infer_action(self, **kwargs):
            return {"action": np.zeros((kwargs["action_horizon"], 7)).view(FakeModel.Out),
                    "force_future_newton": np.zeros((horizon, fdim))}

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
        rec["scene_key_at_end_of_episode"] = "fixture-scene-moved"
        rec["force_steps_scored"] = 0
        before = set(rec)
        out = SM.summarize_episode(rec, calls, clog, chunks)
    except Exception as exc:                                     # noqa: BLE001
        err = "%s: %s" % (type(exc).__name__, exc)

    tail = sm.split('res["episodes"] = episodes')[-1]
    # `(?!\s*=[^=])` skips an assignment, including one written with a space before the `=`, while
    # still counting a read that is compared with `==` or `!=` -- the name list is supposed to be
    # what the tail *consumes*, and a producer it only writes belongs to the fold, not to it.
    consumed = (set(re.findall(r'\br\["([a-z0-9_]+)"\](?!\s*=[^=])', tail))
                | set(re.findall(r'\brec\["([a-z0-9_]+)"\](?!\s*=[^=])', tail)))
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


def check_proprio_attribution(results):
    """A proprio value at the +-5 clamp has to name its column, its raw value and its frame.

    The first live smoke of the merged 40k node reported `proprio_base_min_all = -5.0` and
    `proprio_base_max_all = +5.0` with nothing else attached, which is equally consistent with the
    harness normalising the wrong numbers and with the robot driving the end-effector somewhere the
    42,681 training rows never were -- the two look identical in the aggregate, and only the raw
    column at the same replan separates them. `min/max` mode maps two band widths below `lo` onto
    -5 and two above `hi` onto +5, so the physics is checkable arithmetically; what was missing was
    the recorder keeping the raw observation next to the vector it fed the model.

    Three things are therefore driven through the *real* recorders here, on synthetic observations,
    because a real rollout costs a model load and a GPU minute per case: a far-out raw that must
    land exactly on the clamp and be called explained, a fed vector that reaches the clamp with a
    raw inside the band, which must be called unexplained, and a model that writes into the
    `proprio` tensor it was handed, which must not move the recorded "before" numbers. The last one
    is the reason the snapshot is taken before the call *and copied*: `_as_np` shares memory with a
    CPU float32 tensor, so a copy is the difference between a measurement and a reflection.
    """
    import eval40k as E
    import smoke40k as SM
    import torch

    horizon, fdim, ep_seed = 32, len(CHANNELS), 11
    lo = np.array([0.05462999641895294, 0.09031549096107483, 0.7638311982154846,
                   -1.2248071432113647, -2.4337832927703857, -1.2333282232284546,
                   0.012251690961420536, -0.04023492708802223,
                   -0.7338846325874329, -0.5887312889099121, -0.18089430034160614,
                   -0.6814043521881104, -0.6854666471481323, -1.3969910144805908])
    hi = np.array([0.13350403308868408, 0.3028571903705597, 0.919999361038208,
                   2.480116605758667, 1.233620047569275, 2.443288803100586,
                   0.04023492708802223, -0.012251690961420536,
                   0.7338846325874329, 0.5887312889099121, 0.18089430034160614,
                   0.6814043521881104, 0.6854666471481323, 1.3969910144805908])
    mid = (lo + hi) / 2.0
    rows = [mid.copy(), mid.copy(), mid.copy(), mid.copy()]
    rows[1][0] = lo[0] - 2.5 * (hi[0] - lo[0])      # 2.5 widths below: maps to -6, clamps to -5
    rows[2][2] = hi[2] + 2.5 * (hi[2] - lo[2])      # and above: maps to +6, clamps to +5
    rows[3][9] = lo[9] - 3.0 * (hi[9] - lo[9])      # a force column, where the clamp is normal

    def mapped(row):
        return 2.0 * (np.asarray(row) - lo) / (hi - lo) - 1.0

    before = dict(E.STATS_FACTS)
    E.STATS_FACTS.clear()
    E.STATS_FACTS.update({"source": "flow_test fixture band",
                          "bands": {"default": {"min": lo.tolist(), "max": hi.tolist()}}})

    class Out(np.ndarray):
        requires_grad = False

    def drive(fed_rows, raw_rows, mutate_last=False):
        calls, clog = [], []

        class Model:
            def infer_action(self, **kw):
                if mutate_last and len(calls) == len(fed_rows) - 1:
                    kw["proprio"].mul_(0.0).add_(99.0)     # a sampler writing into its input
                return {"action": np.zeros((horizon, 7), dtype=np.float32).view(Out),
                        "force_future_newton": np.zeros((horizon, fdim), dtype=np.float32).view(Out)}

        class Policy:
            def chunk(self, obs, frame, seed):
                i = len(clog)
                actions = [np.zeros(14) for _ in range(horizon)]
                meta = {"infer_s": 1.0, "replan_from_frame": int(10 * i),
                        "noise_seed": int(noise_seed(ep_seed, i + 1)), "action_absmax": 0.5,
                        "force_feedback_applied": False, "history_mask_true": HIST_LEN}
                return actions, np.zeros((horizon, fdim)), meta

        def witness(obs):
            raw = np.asarray(raw_rows[int(obs["frame"]) // 10], dtype=np.float64)
            return {"raw14": raw.tolist(), "mapped_no_clamp14": mapped(raw).tolist()}

        model, policy = Model(), Policy()
        SM.install_recorder(model, calls)
        SM.install_chunk_recorder(policy, clog, witness=witness)
        chunks = []
        for i, fed in enumerate(fed_rows):
            obs = {"frame": 10 * i}
            tensor = torch.tensor(np.asarray(fed, dtype=np.float32)[None, :])
            model.infer_action(seed=noise_seed(ep_seed, i + 1), rand_device="cpu",
                               num_inference_steps=10, action_horizon=horizon,
                               num_video_frames=9, input_image=np.zeros((1, 3, 4, 4)),
                               proprio=tensor, force_history=np.zeros((1, HIST_LEN, fdim)),
                               force_history_mask=np.ones((1, HIST_LEN), dtype=np.int64) * HIST_LEN)
            actions, pred_force, meta = policy.chunk(obs, frame=10 * i, seed=ep_seed)
            chunks.append({"frame": 10 * i, "pred_force": pred_force, "n_actions": len(actions),
                           "meta": meta})
        rec = SM.summarize_episode({"seed": ep_seed}, calls, clog, chunks)
        return rec, calls, clog

    try:
        fed_ok = [np.clip(mapped(r), -5.0, 5.0) for r in rows]
        rec, calls, clog = drive(fed_ok, rows)
        roll = SM.proprio_band_rollup([rec])
        cols = {e["column"]: e for e in roll["exceedances"]}
        forged_row = rows[0].copy()
        forged_fed = np.clip(mapped(forged_row), -5.0, 5.0)
        forged_fed[4] = -5.0                       # at the clamp, with a raw exactly mid-band
        rec_forged, _c, _l = drive([forged_fed], [forged_row])
        roll_forged = SM.proprio_band_rollup([rec_forged])
        rec_mut, calls_mut, _cl_mut = drive(fed_ok[:1], rows[:1], mutate_last=True)
        ood = SM.episode_band_finding(rec)
        ood_forged = SM.episode_band_finding(rec_forged)
    finally:
        E.STATS_FACTS.clear()
        E.STATS_FACTS.update(before)

    add = lambda desc, ok: results["checks"].append((desc, ok))  # noqa: E731
    results["proprio_attribution"] = {
        "pairs": rec["proprio_band_map_pairs"], "unpaired": rec["proprio_band_map_unpaired"],
        "maxdiff": rec["proprio_band_map_maxdiff"], "exceedance_columns": sorted(cols),
        "unexplained": roll["unexplained"], "forged_unexplained": roll_forged["unexplained"],
        "forged_maxdiff": rec_forged["proprio_band_map_maxdiff"],
        "col0_raw_min": None if 0 not in cols else cols[0]["raw_min"],
        "col0_widths_below": None if 0 not in cols else cols[0]["raw_min_below_band_lo_by_widths"],
        "mutated_calls": rec_mut["proprio_changed_by_model_calls"],
        "mutated_recorded_max": rec_mut["proprio_max_all"],
        "col_min": rec["proprio_col_min"], "raw_min_frame": rec["proprio_raw_min_frame"]}
    add("the fold pairs every replan's fed vector with that replan's own raw observation: %d/%d "
        "paired, %d unpaired, worst element diff %s"
        % (rec["proprio_band_map_pairs"], len(calls), rec["proprio_band_map_unpaired"],
           rec["proprio_band_map_maxdiff"]),
        rec["proprio_band_map_pairs"] == len(calls) == len(clog)
        and rec["proprio_band_map_unpaired"] == 0
        and rec["proprio_band_map_maxdiff"] is not None
        and rec["proprio_band_map_maxdiff"] < 1e-5)
    add("a raw 2.5 band widths below the recorded edge lands exactly on the -5 clamp, and the "
        "rollup names the column, the raw value, the frame and the excursion",
        roll["unexplained"] == 0 and set(cols) == {0, 2, 9}
        and cols[0]["normalised_min"] == -5.0 and cols[0]["raw_min_frame"] == 10
        and cols[0]["raw_min_below_band_lo_by_widths"] == 2.5)
    add("the +5 side is the mirror image: 2.5 widths above `hi` on the third column, attributed "
        "to the frame it was measured at",
        cols[2]["normalised_max"] == 5.0 and cols[2]["raw_max_frame"] == 20
        and cols[2]["raw_max_above_band_hi_by_widths"] == 2.5)
    add("a force column at the clamp is still called explained -- the clamp is part of the "
        "training contract, not damage to be corrected", cols[9]["explained_by_raw_excursion"])
    add("a vector that reaches the clamp while its own raw value sits mid-band is reported as "
        "UNEXPLAINED (%d) and the pairing catches it as a %s mismatch, so a harness bug cannot "
        "hide behind the band check" % (roll_forged["unexplained"], rec_forged["proprio_band_map_maxdiff"]),
        roll_forged["unexplained"] == 1 and rec_forged["proprio_band_map_maxdiff"] > 1e-5)
    add("the snapshot is taken before the call and copied, so a model that writes 99.0 into the "
        "proprio buffer it was handed is recorded as having changed its input (%d call) and the "
        "number reported is still the one fed (%s, not 99.0)"
        % (rec_mut["proprio_changed_by_model_calls"], rec_mut["proprio_max_all"]),
        rec_mut["proprio_changed_by_model_calls"] == 1
        and abs(rec_mut["proprio_max_all"] - float(np.clip(mapped(rows[0]), -5, 5).max())) < 1e-6)
    add("the per-episode check the drivers run refuses a record whose pairing did not survive: "
        "the assertion is in the source of both drivers, not only here",
        'rec["proprio_band_map_pairs"] == n' in (HERE / "smoke40k.py").read_text(encoding="utf-8")
        and "proprio_band_rollup(records)" in (HERE / "eval40k.py").read_text(encoding="utf-8"))
    results["episode_band_finding"] = {
        "base_out": ood["base_columns_out_of_band"], "force_out": ood["force_columns_out_of_band"],
        "worst_widths": ood["worst_base_excursion_band_widths"],
        "unexplained": ood["unexplained"], "clamped_calls": ood["clamped_calls"],
        "forged_unexplained": ood_forged["unexplained"],
        "forged_any_unexplained": ood_forged["any_excursion_unexplained"]}
    add("the per-episode finding separates the two questions the aggregate band number used to "
        "blur: it reports base columns %s and force columns %s as having left the recorded band, "
        "%s band widths the worst, from %d clamped calls, and calls the episode explained (%d) "
        "because each excursion names its own raw value and frame"
        % (ood["base_columns_out_of_band"], ood["force_columns_out_of_band"],
           ood["worst_base_excursion_band_widths"], ood["clamped_calls"], ood["unexplained"]),
        ood["base_columns_out_of_band"] == [0, 2] and ood["force_columns_out_of_band"] == [9]
        and ood["worst_base_excursion_band_widths"] == 2.5
        and ood["clamped_calls"] == 3 and ood["unexplained"] == 0
        and ood["any_excursion_unexplained"] is False
        and ood["episodes_without_witness"] == [])
    add("the same finding on a vector that reached the clamp while its raw value stayed mid-band "
        "reports %d unexplained and any_excursion_unexplained=%s, so the new per-episode check "
        "fails on a harness bug rather than on the robot moving"
        % (ood_forged["unexplained"], ood_forged["any_excursion_unexplained"]),
        ood_forged["unexplained"] == 1 and ood_forged["any_excursion_unexplained"] is True)


def check_inference_contract(results):
    """The three failures the first live rollout blamed on the model, tested where they happened.

    `step_040000` ran two real episodes and then reported that the model's device did not match the
    policy's, that its chunk shapes were wrong and that the two episodes started from two different
    scenes. All three were the harness's own expectations: the parameters were on `cuda:0` while the
    config says `cuda` (the same hardware, spelled differently), the model returns `[T, D]` because
    it indexes `latents[0]` on the way out (the smoke expected `[1, T, D]`), and the scene hash was
    taken after the robot had moved the cookies (so one fixed layout produced two hashes). Each one
    cost a model load and a GPU minute to discover, which is exactly what this file exists to
    prevent, so each is pinned here at the level the mistake was made at.

    The shape convention is read out of the model's own source, because an expectation written from
    a memory of what the tensor looked like is what broke; the device rule is run against real
    `torch.device` objects, including the pair the live check compares; the scene assertion is about
    *where in the driver* the hash is taken, since no synthetic environment can be moved by a robot;
    and the wall-time split is driven through the real meter on a fake environment whose step cost
    is written down here, because the 160-episode batch is sized from those numbers and one
    CPU-starved episode in the previous run made the mean projection useless.
    """
    import time

    import eval40k as E
    import smoke40k as SM
    import torch

    add = lambda desc, ok: results["checks"].append((desc, ok))  # noqa: E731
    sm = (HERE / "smoke40k.py").read_text(encoding="utf-8")
    ev = (HERE / "eval40k.py").read_text(encoding="utf-8")
    horizon, adim, fdim = 32, 7, len(CHANNELS)

    # --- the shape the model is contracted to hand back ----------------------------------------
    wan = HERE.parent.parent / "fw" / "src" / "fasterwam" / "models" / "wan22"
    jw_path, fw_path = wan / "jointwam.py", wan / "fastwam.py"
    jw = jw_path.read_text(encoding="utf-8") if jw_path.is_file() else ""
    fsrc = fw_path.read_text(encoding="utf-8") if fw_path.is_file() else ""
    want_action, want_force = SM.model_chunk_shapes(horizon, adim, fdim)
    add("the contract the drivers check against is two-dimensional -- %s for the action, %s for "
        "the predicted force -- not the (1, T, D) the first live smoke expected"
        % (want_action, want_force),
        want_action == [horizon, adim] and want_force == [horizon, fdim]
        and len(want_action) == len(want_force) == 2)
    add("and the model really does drop the batch dimension where those two numbers come from: "
        "every `action` entry in %s (both diffusion paths) is built as `latents_action[0]`, and the "
        "force payload in %s is built as `raw[0]` of the denormalised `state[\"latents\"]` -- read "
        "off the source the checkpoint was trained against, so the convention cannot silently "
        "change under this file" % (jw_path.name, fw_path.name),
        bool(jw) and bool(fsrc)
        and jw.count('"action": latents_action[0]') == jw.count('"action": latents_action') >= 2
        and fsrc.count('"force_future_newton": raw[0]')
        == fsrc.count('"force_future_newton":') >= 1
        and fsrc.count('"force_future": state["latents"][0]')
        == fsrc.count('"force_future":') >= 1)

    # --- the recorder, driven with each shape, reports what the smoke's check reads -------------
    def drive(action_shape, force_shape):
        calls, clog = [], []

        class Out(np.ndarray):
            requires_grad = False

        class Model:
            def infer_action(self, **kw):
                return {"action": np.zeros(action_shape, dtype=np.float32).view(Out),
                        "force_future_newton": np.zeros(force_shape, dtype=np.float32).view(Out)}

        class Policy:
            def chunk(self, obs, frame=None, seed=None):
                meta = {"infer_s": 2.0, "replan_from_frame": int(frame or 0), "noise_seed": 7,
                        "action_absmax": 0.5}
                return ([np.zeros(adim) for _ in range(horizon)],
                        np.zeros((horizon, fdim), dtype=np.float32), meta)

        model, policy = Model(), Policy()
        SM.install_recorder(model, calls)
        SM.install_chunk_recorder(policy, clog)
        model.infer_action(seed=7, rand_device="cpu", num_inference_steps=10,
                           action_horizon=horizon, num_video_frames=9,
                           input_image=np.zeros((1, 3, 4, 4)),
                           proprio=torch.from_numpy(np.zeros((1, E.PROPRIO_DIM), dtype=np.float32)),
                           force_history=np.zeros((1, HIST_LEN, fdim), dtype=np.float32),
                           force_history_mask=np.ones((1, HIST_LEN), dtype=np.int64) * HIST_LEN)
        actions, pred_force, meta = policy.chunk({"frame": 0}, frame=0, seed=11)
        chunks = [{"frame": 0, "pred_force": pred_force, "n_actions": len(actions), "meta": meta}]
        return SM.summarize_episode({"seed": 11, "decision_chunks": 1}, calls, clog, chunks)

    rec2d, rec3d = drive(want_action, want_force), drive([1] + want_action, [1] + want_force)
    add("folded through the real recorders, a %s model output is what the two shape fields say it "
        "is: `action_shape_reported` %s, `force_shape_reported` %s and the de-duplicated "
        "`model_action_shapes` %s" % (want_action, rec2d["action_shape_reported"],
                                      rec2d["force_shape_reported"], rec2d["model_action_shapes"]),
        rec2d["action_shape_reported"] == want_action
        and rec2d["force_shape_reported"] == want_force
        and rec2d["model_action_shapes"] == [str(want_action)])
    add("the same fold refuses the batched shape rather than agreeing with it: a (1, %d, %d) action "
        "is reported as %s, which is not %s, so the corrected expectation has teeth and was not "
        "written to match whatever the smoke happened to print"
        % (horizon, adim, rec3d["model_action_shapes"], want_action),
        rec3d["action_shape_reported"] == [1, horizon, adim]
        and rec3d["force_shape_reported"] == [1, horizon, fdim]
        and rec3d["model_action_shapes"] != [str(want_action)]
        and all(r["chunk_shape"] == [horizon, adim] for r in rec3d["chunk_log"]))

    # --- devices: `cuda` and `cuda:0` are one piece of hardware --------------------------------
    cur = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    dev_cases = [("cpu", "cpu", True),
                 ("cpu", "cuda:%d" % cur, False),
                 ("cuda:%d" % cur, "cuda:%d" % (cur + 1), False),
                 ("cuda", "cuda:%d" % (cur + 1), False),
                 ("cuda", "cuda:%d" % cur, bool(torch.cuda.is_available()))]
    dev_got = [(a, b, SM.same_device(a, b)) for a, b, _e in dev_cases]
    results["device_contract"] = {
        "cuda_available": bool(torch.cuda.is_available()), "current_device": cur,
        "cases": [[str(a), str(b), bool(ok)] for a, b, ok in dev_got],
        "string_spelling_of_current": str(torch.device("cuda:%d" % cur)),
        "spelled_in_config": "cuda"}
    add("the device rule resolves the index instead of comparing strings: cpu is cpu, cpu is not "
        "cuda, a bare `cuda` names the current device (%d) and not its neighbour (%d), and it only "
        "matches `cuda:%d` where CUDA is actually present -- %s"
        % (cur, cur + 1, cur, results["device_contract"]["cases"]),
        all(ok == exp for (_a, _b, ok), (_a2, _b2, exp) in zip(dev_got, dev_cases)))
    add("the pair the live check compares is spelled differently but is one device: %s is not the "
        "string `cuda`, so the `str() ==` comparison the first smoke used could only ever report a "
        "mismatch, and the source no longer contains it"
        % results["device_contract"]["string_spelling_of_current"],
        str(torch.device("cuda:%d" % cur)) != "cuda"
        and '== str(policy.device)' not in sm
        and 'same_device(next(policy.model.parameters()).device, policy.device)' in sm)

    # --- scenes: the hash has to be taken before the robot touches the cookies ------------------
    add("the fixed-scene check hashes the layout inside `reset`, where the cookies are still where "
        "the scene put them, and the episode record reads that value back: %s"
        % "scene_at_reset",
        source_before(sm, 'scene_at_reset[int(seed)] = E.scene_key(env)',
                      'run_episode(env, policy, seed=seed')
        and 'rec["scene_key"] = scene_at_reset.get(' in sm)
    add("and the end-of-episode pose is kept in a field with a different name, so the mistake "
        "cannot come back by reusing the variable: no `rec[\"scene_key\"] = E.scene_key(` is left, "
        "and the batch driver records its scene the same way at its own reset",
        'rec["scene_key"] = E.scene_key(' not in sm
        and 'rec["scene_key_at_end_of_episode"] = E.scene_key(env)' in sm
        and source_before(ev, 'scene_keys[int(seed)] = scene_key(env)',
                          'rec, measured, chunks = run_episode'))

    # --- time: simulation, inference, and what the machine owes us ------------------------------
    class Tick:
        """A stand-in environment whose per-step cost is known before it runs."""

        def __init__(self, delay):
            self.delay, self.returns, self.i = delay, [], 0

        def step(self, action):
            time.sleep(self.delay)
            self.i += 1
            self.returns.append({"tick": self.i, "action": action})
            return self.returns[-1]

    delay, n_steps = 0.02, 5
    tick = Tick(delay)
    meter = SM.install_env_timer(tick)
    outs = [tick.step(i) for i in range(n_steps)]
    add("the step meter counts one tick per simulator step (%d) and hands the step's own return "
        "value back unchanged, so wrapping the environment cannot alter the rollout it is "
        "measuring -- and it cannot lose a step either, which is what would make an 8x episode "
        "look like inference" % meter["steps"],
        meter["steps"] == n_steps and [o["action"] for o in outs] == list(range(n_steps))
        and outs == tick.returns
        and meter["s_total"] >= n_steps * delay
        and delay <= meter["s_max"] <= meter["s_total"])

    rt = SM.record_time_split({"wall_s": 100.0, "infer_s_total": 30.0},
                              {"steps": 7, "s_total": 60.0, "s_max": 0.5},
                              {"throttled_usec": 1000000, "usage_usec": 5000000},
                              {"throttled_usec": 4000000, "usage_usec": 9000000})
    rt_na = SM.record_time_split({"wall_s": 50.0, "infer_s_total": 10.0},
                                 {"steps": 3, "s_total": 40.0, "s_max": 0.25}, {}, {})
    add("an episode's wall time is split into simulation (%s s of %s s), an inference share of %s "
        "and a named remainder (%s s), with the throttling counters carried as deltas; with no "
        "counters readable the deltas are None rather than a fabricated zero: %s"
        % (rt["sim_s_total"], rt["wall_s"], rt["infer_s_share"], rt["unaccounted_s"],
           (rt["cpu_throttled_usec_delta"], rt_na["cpu_throttled_usec_delta"])),
        rt["sim_steps"] == 7 and rt["sim_s_total"] == 60.0 and rt["sim_s_max_step"] == 0.5
        and rt["infer_s_share"] == 0.3 and rt["unaccounted_s"] == 10.0
        and rt["cpu_throttled_usec_delta"] == 3000000
        and rt["cpu_usage_usec_delta"] == 4000000
        and rt_na["unaccounted_s"] == 0.0 and rt_na["cpu_throttled_usec_delta"] is None
        and rt_na["cpu_usage_usec_delta"] is None)

    import ast
    import tempfile

    tmp = Path(tempfile.mkdtemp()) / "cpu.stat"
    tmp.write_text("usage_usec 1234567\nuser_usec 1\nsystem_usec 2\nnr_periods 40\n"
                   "nr_throttled 7\nthrottled_usec 987654\nnr_bursts 0\nburst_usec 5\n")
    facts = SM.cpu_throttle_facts(tmp)
    # `/proc/loadavg` may be named in the prose that explains why it is the wrong number to read;
    # what must not exist is a path the code opens. String constants are the honest way to tell.
    literals = {n.value for n in ast.walk(ast.parse(sm))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    results["cpu_throttle_facts"] = {"parsed": facts, "missing_path":
                                     SM.cpu_throttle_facts(tmp.parent / "absent"),
                                     "path_read": SM.CPU_STAT_PATH}
    add("the container's own cpu.stat is read for the four fields that separate a slow simulator "
        "from a starved one (%s), an unreadable path gives {} rather than a guess, and the host's "
        "/proc/loadavg is never opened (%s)" % (facts, SM.CPU_STAT_PATH),
        facts == {"usage_usec": 1234567, "nr_periods": 40, "nr_throttled": 7,
                  "throttled_usec": 987654}
        and SM.cpu_throttle_facts(tmp.parent / "absent") == {}
        and SM.CPU_STAT_PATH == "/sys/fs/cgroup/cpu.stat"
        and "/proc/loadavg" not in literals)

    starved = [{"seed": 11, "wall_s": 196.0, "sim_s_total": 120.0, "infer_s_total": 60.0,
                "unaccounted_s": 16.0, "cpu_throttled_usec_delta": 1000000},
               {"seed": 12, "wall_s": 1673.0, "sim_s_total": 1590.0, "infer_s_total": 61.0,
                "unaccounted_s": 22.0, "cpu_throttled_usec_delta": 1489000000},
               {"seed": 13, "wall_s": 210.0, "sim_s_total": 130.0, "infer_s_total": 62.0,
                "unaccounted_s": 18.0, "cpu_throttled_usec_delta": 2000000}]
    roll_t = SM.time_rollup(starved)
    results["time_rollup"] = roll_t
    add("the rollup keeps one episode's starvation from rewriting the batch estimate: walls %s give "
        "a median of %s against a mean of %s, inference is flat across the three (%s) while the "
        "simulation column is where the 8x went (%s), and the throttled time is summed in seconds "
        "(%s)"
        % ([e["wall_s"] for e in starved], roll_t["wall"]["median"], roll_t["wall"]["mean"],
           roll_t["infer"], roll_t["sim"], roll_t["cpu_throttled_s_total"]),
        roll_t["wall"]["median"] == 210.0 and roll_t["wall"]["mean"] == 693.0
        and roll_t["wall"]["median"] < roll_t["wall"]["mean"]
        and roll_t["infer"]["mean"] == 61.0 and roll_t["infer"]["median"] == 61.0
        and roll_t["sim"]["median"] == 130.0 and roll_t["sim"]["max"] == 1590.0
        and roll_t["cpu_throttled_s_total"] == 1492.0
        and [r["seed"] for r in roll_t["per_episode_s"]] == [11, 12, 13]
        and [r["cpu_throttled_s"] for r in roll_t["per_episode_s"]] == [1.0, 1489.0, 2.0]
        and SM.time_rollup([])["wall"] is None)
    add("the split is measured on the batch path too, not only in the two-episode smoke: `eval40k` "
        "wraps every real environment in `install_env_timer`, folds the meter into each record and "
        "rolls the node's episodes into `summary[\"timing\"]`, so the 160-episode projection is "
        "made from the batch's own numbers",
        "install_env_timer(env)" in ev and "record_time_split(rec, meter" in ev
        and 'summary["timing"] = timing' in ev and "time_rollup(records)" in ev)
    add("and both statistics of that projection reach the report -- median and mean per 160 "
        "episodes -- next to the per-episode remainder the acceptance checks are written against",
        'res["timing"] = time_rollup(episodes)' in sm
        and "projected_160_episodes_h_median" in sm
        and "projected_160_episodes_h_mean" in sm
        and 'all(r.get("sim_steps") == r.get("steps_executed")' in sm
        and 'timing["projected_160_episodes_h_median"]' in ev
        and 'timing["projected_160_episodes_h_mean"]' in ev)


def check_batch_disk_gate(results):
    """The disk gate the 160-episode run is refused by, tested in the units it actually compares.

    The round-9 launch attempt stopped with `only 27.44 GiB free, merging needs 15.53 GiB at margin
    1.5x` -- a true statement about a volume with 27 GiB free and a 15 GiB requirement, because the
    comparison put `free_gib()`'s GiB reading against `MERGED_BYTES * margin`, which is in bytes. It
    was true for every volume that could hold the run, and it sat behind `if not args.dry_run`, so
    the step-6 interlock printed `blockers: []` four minutes before the real launch refused. These
    checks pin the threshold in GiB, pin that the gate still fires below it, pin that a bigger volume
    is never blocked, and on the pod pin that `--dry-run` now answers the same question the launch
    does.
    """
    add = lambda desc, ok: results["checks"].append((desc, ok))  # noqa: E731
    need_gib = BE.MERGED_BYTES * 1.5 / 2**30
    pod_free_gib = 27.44                       # measured on the run40k volume, 2026-11-12T07:26Z
    out = {"need_gib": round(need_gib, 3), "dry_run_exercised": False}

    add("the merge gate compares GiB with GiB: one artifact of %d B at margin 1.5 is %.2f GiB, and "
        "the volume the batch was planned against (%.2f GiB free) is not blocked by it"
        % (BE.MERGED_BYTES, need_gib, pod_free_gib),
        15.0 < need_gib < 16.0 and BE.disk_blocker(pod_free_gib, BE.MERGED_BYTES, 1.5) is None)
    add("the gate still fires below its threshold and says what it compared: 14 GiB free against "
        "the same requirement yields a blocker naming both numbers in GiB, not a silent pass",
        (BE.disk_blocker(14.0, BE.MERGED_BYTES, 1.5) or "").startswith("only 14.00 GiB free")
        and "15.53 GiB" in BE.disk_blocker(14.0, BE.MERGED_BYTES, 1.5)
        and "1.5x" in BE.disk_blocker(14.0, BE.MERGED_BYTES, 1.5))
    add("and the unit error cannot come back: the pre-fix comparison would have blocked a 1 TiB "
        "volume, so a 1024 GiB reading must pass the corrected gate while `1024.0 < MERGED_BYTES * "
        "1.5` is still true -- the difference between the two forms is the bug, not the threshold",
        BE.disk_blocker(1024.0, BE.MERGED_BYTES, 1.5) is None
        and (1024.0 < BE.MERGED_BYTES * 1.5) is True)

    # End to end through main(), which is the only way to show the dry-run path reports it. Needs the
    # screened seed list, so it runs on the pod; on the stub machine batch_eval exits before it gets
    # to the gate, and saying so here is better than a check that passes without being exercised.
    import contextlib
    import io
    import tempfile
    tmp = tempfile.mkdtemp()
    old_argv, old_free = list(sys.argv), BE.free_gib
    try:
        blockers = {}
        for value in (pod_free_gib, 1.0):
            BE.free_gib = lambda p, v=value: v
            buf = io.StringIO()
            sys.argv = ["batch_eval.py", "--dry-run", "--out", tmp]
            with contextlib.redirect_stdout(buf):
                BE.main()
            doc = json.loads(buf.getvalue())
            blockers[value] = [b for b in doc["blockers"] if "GiB free" in b]
        out["dry_run_exercised"] = True
        out["blockers_by_free_gib"] = {str(k): v for k, v in blockers.items()}
        add("`--dry-run` itself carries the disk gate, so the interlock that reads its blockers can "
            "refuse a run before the launch does: %.2f GiB free gives no disk blocker and 1.00 GiB "
            "gives exactly one (%s)" % (pod_free_gib, blockers[1.0]),
            blockers[pod_free_gib] == [] and len(blockers[1.0]) == 1
            and "15.53 GiB" in blockers[1.0][0])
    except (SystemExit, Exception) as exc:                       # noqa: BLE001
        out["dry_run_skipped_reason"] = str(exc)[:200]
    finally:
        sys.argv = old_argv
        BE.free_gib = old_free
    src = (HERE / "batch_eval.py").read_text(encoding="utf-8")
    add("the call site goes through the one helper rather than repeating the arithmetic, so a future "
        "edit cannot put a byte threshold back next to a GiB reading",
        src.count("disk_blocker(free_gib(merged_dir), MERGED_BYTES, args.disk_margin)") == 1
        and "free_gib(merged_dir) < MERGED_BYTES * args.disk_margin\n" not in src)
    results["batch_disk_gate"] = out
    return out


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
    check_proprio_attribution(results)
    check_inference_contract(results)
    check_batch_disk_gate(results)
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
