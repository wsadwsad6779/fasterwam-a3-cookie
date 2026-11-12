"""CPU self-test for the P5 corrective-data collector: 96 assertions with hand-computed answers.

Every expectation below is derived on paper from the fixture world, not read back from the code
under test -- that is the only way a test of a *recorder* means anything. A recorder is exactly
the component that can pass a test written against itself while quietly storing something else,
so each check states the number of rows, which source owned them, and what action a reader of
the npz would find there.

Run it anywhere numpy runs; it needs no model, no simulator and no GPU:

    cd /workspace/_v1_stage/dev/p5_dagger
    PYTHONDONTWRITEBYTECODE=1 python dagger_selftest.py
    PYTHONDONTWRITEBYTECODE=1 python dagger_selftest.py --json out/dagger_selftest.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "eval40k"))
sys.path.insert(0, str(HERE.parent / "p4_shadow"))

from dagger5 import (Controller, DaggerIntegrityError, ForceZAbove, ManualFrames,  # noqa: E402
                     NoProgress, NonfiniteState, ShadowVerdict, StepTable,
                     labels_for_training, run_dagger_episode, verify_rows)
from force_stream import CHANNELS, HIST_LEN, ForceStream, select_channels   # noqa: E402

ACTION_DIM = 14
FORCE_DIM = 18
HORIZON = 32
EP_LEN = 25                      # frames 0..24; frame 24 is the terminal observation
N_STEPS = EP_LEN - 1             # 24 executed steps, so a clean episode has 24 rows
COOKIE_FRAMES = (7, 19)
CALIB_MEAN = np.zeros(6)
CALIB_STD = np.ones(6)

RESULTS = {"checks": [], "n_checks": 0, "n_passed": 0, "failed": []}


def check(name, cond, detail=None):
    ok = bool(cond)
    RESULTS["checks"].append({"check": name, "pass": ok, "detail": detail})
    RESULTS["n_checks"] += 1
    RESULTS["n_passed"] += int(ok)
    if not ok:
        RESULTS["failed"].append(name)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  <- {detail}"), flush=True)


# --------------------------------------------------------------------------- fixture world

def frozen_force(frame):
    """Six frozen channels: fx grows 0.5 N/frame, fz is flat, gripper is 2 N."""
    return np.array([0.5 * frame, -0.25 * frame, 1.0, 0.0, 0.0, 2.0], dtype=np.float64)


def force18(frame, nan=False):
    v = np.zeros(FORCE_DIM, dtype=np.float64)
    v[list(CHANNELS)] = frozen_force(frame)
    if nan:
        v[CHANNELS[0]] = np.nan
    return v


class DaggerFixtureEnv:
    """Deterministic stand-in: one observation per frame, `applied_action` echoes its input."""

    def __init__(self, ep_len=EP_LEN, nan_force_frame=None, nan_eef_frame=None,
                 cookie_frames=COOKIE_FRAMES):
        self.ep_len = int(ep_len)
        self.nan_force_frame = None if nan_force_frame is None else int(nan_force_frame)
        self.nan_eef_frame = None if nan_eef_frame is None else int(nan_eef_frame)
        self.cookie_frames = tuple(cookie_frames)
        self.frame = 0
        self.steps = 0

    def _obs(self):
        f = self.frame
        eef = np.array([0.1, 0.2, 0.3 - 0.01 * f, 1, 0, 0, 0])
        if self.nan_eef_frame == f:
            eef[2] = np.nan
        return {"observation.force": force18(f, nan=(self.nan_force_frame == f)),
                "observation.eef_pose": eef,
                "observation.state": np.concatenate([np.zeros(7), [0.001 * f], np.zeros(6)])}

    def reset(self, seed):
        self.frame = 0
        self.steps = 0
        self.seed = int(seed)
        return self._obs()

    def step(self, action):
        self.frame += 1
        self.steps += 1
        applied = np.asarray(action, dtype=np.float64).copy()
        cin = sum(1 for c in self.cookie_frames if c <= self.frame)
        info = {"applied_action": applied, "cookies_in_target": cin, "step": self.frame}
        terminated = self.frame >= self.ep_len - 1
        return self._obs(), (1.0 if terminated else 0.0), terminated, False, info

    def close(self):
        pass


class CountingPolicy:
    """Emits actions it can be audited against: `a[f][k] = 0.001*f + 0.0001*k`."""

    def __init__(self, horizon=HORIZON, n_actions=10):
        self.horizon = int(horizon)
        self.n_actions = int(n_actions)
        self.stream = ForceStream(hist_len=HIST_LEN)
        self.chunk_calls = []
        self.observe_calls = 0
        self.emitted = {}

    def start_episode(self, obs, seed):
        self.stream.reset()
        self.chunk_calls = []
        self.observe_calls = 0
        self.emitted = {}

    def observe(self, obs, frame):
        if self.stream.n_frames != frame:
            raise RuntimeError(f"policy stream has {self.stream.n_frames} frames at frame {frame}")
        self.stream.push(obs["observation.force"])
        self.observe_calls += 1

    def chunk(self, obs, frame, seed):
        self.chunk_calls.append(int(frame))
        actions = []
        for k in range(self.n_actions):
            a = np.full(ACTION_DIM, 0.001 * frame + 0.0001 * k)
            actions.append(a)
            self.emitted[(int(frame), k)] = a
        # pred[j] == j makes the horizon alignment of the recorded rows checkable by eye
        pf = np.tile(np.arange(self.horizon, dtype=np.float64)[:, None], (1, 6))
        return actions, pf, {"chunk_frame": int(frame)}


class StatefulExpert:
    """Like `A3SameColumnBatchExpert`: `act()` advances an internal phase machine."""

    stateless_peek = False

    def __init__(self, dim=ACTION_DIM):
        self.dim = int(dim)
        self.phase = 0
        self.phase_steps = 0
        self.act_calls = 0

    def reset(self):
        self.phase = 0
        self.phase_steps = 0
        self.act_calls = 0

    def act(self, obs):
        self.act_calls += 1
        self.phase_steps += 1
        if self.phase_steps == 5:
            self.phase += 1
            self.phase_steps = 0
        return np.full(self.dim, 0.5 + 0.1 * self.phase)


class StatelessExpert(StatefulExpert):
    """An expert whose action depends only on the observation, so peeking cannot corrupt it."""

    stateless_peek = True

    def act(self, obs):
        self.act_calls += 1
        f = np.asarray(obs["observation.force"], dtype=np.float64)
        return np.full(self.dim, float(f[CHANNELS[0]]) * 0.01)


class FakeShadow:
    """Stands in for `ShadowMonitor` when only a verdict stream is wanted."""

    def __init__(self, verdicts=("proceed",)):
        self.verdicts = list(verdicts)
        self.calls = []
        self._i = 0

    def on_decision(self, payload):
        self.calls.append(int(payload["frame"]))
        v = self.verdicts[min(self._i, len(self.verdicts) - 1)]
        self._i += 1
        return {"verdict": v}

    def flush(self, seed, measured):
        return 0


def run(policy=None, expert=None, env=None, controller=None, shadow=None, seed=7,
        peek_expert=False, **kw):
    env = env or DaggerFixtureEnv()
    policy = policy or CountingPolicy()
    expert = expert or StatefulExpert()
    controller = controller or Controller(rules=())
    report, table, measured = run_dagger_episode(
        env, policy, expert, lambda e, s: e.reset(s), seed,
        max_steps=kw.pop("max_steps", 200), replan=kw.pop("replan", 10),
        controller=controller, shadow=shadow, peek_expert=peek_expert,
        action_dim=ACTION_DIM, **kw)
    return report, table, measured, env, policy, expert, controller, shadow


# --------------------------------------------------------------------------- the tests

def t_clean_episode():
    report, table, measured, env, policy, expert, _c, _s = run()
    rows = table.rows
    check("a rule-free episode records one row per executed step, not per chunk",
          len(rows) == N_STEPS, {"rows": len(rows), "want": N_STEPS})
    check("and every one of them is policy-controlled",
          all(r["source"] == "policy" for r in rows), None)
    check("with zero intervention flags and zero trainable labels",
          report["intervention_rows"] == 0 and report["trainable_label_rows"] == 0, report)
    check("the policy was asked for a chunk exactly at the replan boundaries 0,10,20",
          policy.chunk_calls == [0, 10, 20], policy.chunk_calls)
    check("and observed every single frame, takeover or not (24 steps -> 1+24 pushes)",
          policy.observe_calls == N_STEPS + 1, policy.observe_calls)
    check("the force stream advanced one frame per executed step",
          policy.stream.n_frames == N_STEPS + 1, policy.stream.n_frames)
    check("one state row per visited frame, i.e. steps + 1",
          report["state_rows"] == N_STEPS + 1 and report["integrity_ok"], report)
    check("every row's policy action is the array the policy says it emitted for that frame",
          all(np.array_equal(r["policy_action"],
                             policy.emitted[(r["frame"] - r["frame"] % 10, r["frame"] % 10)])
              for r in rows), None)
    check("and the applied action the env reported matches what was executed",
          all(np.array_equal(r["applied_action"], r["executed_action"]) for r in rows), None)
    check("the recorded force6 is exactly the frozen channels of the recorded force18",
          all(np.allclose(r["force6"], r["force18"][list(CHANNELS)]) for r in rows), None)
    check("force18 at row f is the observation the action was taken from, not the next one",
          np.allclose(rows[5]["force18"], force18(5))
          and np.allclose(rows[5]["force6"], frozen_force(5)), rows[5]["force18"].tolist())
    check("cookies_in_target follows the env's own counter at the post-step frame",
          [r["cookies_in_target"] for r in rows] ==
          [sum(1 for c in COOKIE_FRAMES if c <= f + 1) for f in range(N_STEPS)], None)


def t_pred_alignment():
    _report, table, _m, _e, _p, _x, _c, _s = run()
    rows = table.rows
    want = [float(f % 10) for f in range(N_STEPS)]
    got = [float(r["pred_force"][0]) for r in rows]
    check("the predicted force stored on a row is the chunk's entry for the NEXT frame",
          got == want, {"got": got[:12], "want": want[:12]})
    check("so a chunk issued at frame 0 covers frames 1..10 with indices 0..9",
          float(rows[0]["pred_force"][0]) == 0.0 and float(rows[9]["pred_force"][0]) == 9.0, None)


def t_manual_takeover_sticky_end():
    ctrl = Controller(rules=[ManualFrames([(7, 8)])], sticky="end")
    report, table, _m, _e, policy, expert, _c, _s = run(controller=ctrl)
    rows = table.rows
    n_pol, n_exp = 8, N_STEPS - 8
    check("a manual takeover at frame 8 makes frames 0..7 policy and 8..23 expert",
          [r["source"] for r in rows[:n_pol]] == ["policy"] * n_pol
          and set(r["source"] for r in rows[n_pol:]) == {"expert"}, None)
    check("the split is exactly 8 and 16 rows",
          report["by_source"]["policy"] == n_pol
          and report["by_source"]["expert"] == n_exp, report["by_source"])
    check("the two leftover chunk actions are reported as discarded, not silently dropped",
          report["discarded_policy_actions"] == 2, report)
    check("the expert is asked once per expert row and no more",
          expert.act_calls == n_exp, expert.act_calls)
    check("the model is never asked for a chunk while the expert holds the wheel",
          policy.chunk_calls == [0], policy.chunk_calls)
    check("every expert row is labelled executed, and only those are trainable",
          report["by_label_kind"]["executed"] == n_exp
          and report["trainable_label_rows"] == n_exp, report["by_label_kind"])
    check("the reason recorded on the takeover frame names the rule and the frame",
          rows[8]["reason"] == "manual:f8", rows[8]["reason"])
    check("and the frames after it name the stickiness, not a second trigger",
          rows[9]["reason"] == "sticky:end", rows[9]["reason"])
    check("the expert action actually reached the simulator",
          all(np.array_equal(r["executed_action"], np.full(ACTION_DIM, 0.5))
              for r in rows[n_pol:n_pol + 4]), None)
    check("intervention flag equals expert control on all 24 rows",
          report["integrity_ok"] and all(bool(r["intervention"]) == (r["source"] == "expert")
                                          for r in rows), report["integrity_problems"])


def t_sticky_rule_and_release_after():
    ctrl = Controller(rules=[ManualFrames([(7, 5), (7, 15)])], sticky="rule")
    report, table, _m, _e, policy, expert, _c, _s = run(controller=ctrl)
    rows = table.rows
    check("with sticky=rule two isolated manual points give exactly two expert rows",
          [r["frame"] for r in rows if r["source"] == "expert"] == [5, 15], None)
    check("and the policy is asked again after the release (chunks 0,6,16)",
          policy.chunk_calls == [0, 6, 16], policy.chunk_calls)
    check("both handovers drop the model's unused chunk actions, and say how many",
          report["discarded_policy_actions"] == 5 + 1, report)

    ctrl2 = Controller(rules=[ManualFrames([(7, 5)])], sticky="rule", release_after=3)
    report2, table2, *_rest = run(controller=ctrl2)
    rows2 = table2.rows
    check("release_after=3 holds the wheel for frames 5,6,7 only",
          [r["frame"] for r in rows2 if r["source"] == "expert"] == [5, 6, 7], None)
    check("with the trigger frame plus two sticky frames",
          [rows2[5]["reason"], rows2[6]["reason"], rows2[7]["reason"]] ==
          ["manual:f5", "sticky:1f_of3", "sticky:2f_of3"],
          [rows2[5]["reason"], rows2[6]["reason"], rows2[7]["reason"]])
    check("and control returns to the model on frame 8",
          rows2[8]["source"] == "policy" and rows2[8]["reason"] == "policy", None)
    check("the expert saw 3 frames while the policy resumed for the rest",
          report2["by_source"]["expert"] == 3 and report2["by_source"]["policy"] == N_STEPS - 3,
          report2["by_source"])


def t_force_z_rule():
    # fx = 0.5*frame, mean 0, std 1 -> z = 0.5*frame, so z > 5 from frame 11 onward
    ctrl = Controller(rules=[ForceZAbove(CALIB_MEAN, CALIB_STD, z=5.0)], sticky="end")
    report, table, _m, _e, policy, expert, _c, _s = run(controller=ctrl)
    rows = table.rows
    check("a force-z rule fires on the first frame whose z exceeds the band (frame 11)",
          rows[10]["source"] == "policy" and rows[11]["source"] == "expert", None)
    check("and the reason carries the channel and the measured z",
          rows[11]["reason"].startswith("force_z:c0=z"), rows[11]["reason"])
    check("the takeover covers the rest of the episode",
          report["by_source"]["expert"] == N_STEPS - 11, report["by_source"])
    check("the rule's thresholds are written into the report, not only into the code",
          report["controller"]["rules"][0]["z"] == 5.0
          and report["controller"]["sticky"] == "end", report["controller"])
    try:
        ForceZAbove(np.zeros(6), np.zeros(6), z=5.0)
        ok = False
    except DaggerIntegrityError:
        ok = True
    check("a zero-std calibration is refused by the rule instead of becoming an infinite z", ok)


def t_no_progress_rule():
    # cookies land at frames 7 and 19; patience 8 from frame 8 gives a takeover at frame 16
    ctrl = Controller(rules=[NoProgress(patience=8)], sticky="rule")
    _report, table, _m, _e, _p, _x, _c, _s = run(controller=ctrl)
    rows = table.rows
    first = [r["frame"] for r in rows if r["source"] == "expert"]
    check("no-progress counts from the first frame that raised the task counter",
          first and first[0] == 15, first[:3])
    check("and the reason states the counter and how long it sat",
          rows[first[0]]["reason"].startswith("no_progress:cookies_in_target=1x"),
          rows[first[0]]["reason"])


def t_nan_force_break_path():
    env = DaggerFixtureEnv(nan_force_frame=12)
    report, table, measured, _e, _p, _x, _c, _s = run(env=env)
    rows = table.rows
    check("the step that produced a non-finite force is recorded rather than lost",
          len(rows) == 12 and rows[11]["note"] == "nonfinite_force_after_step", len(rows))
    check("and the loop reports where it aborted instead of pretending the episode ended",
          report["nan_abort_frame"] == 12, report["nan_abort_frame"])
    check("the measured history stops before the bad frame, so nothing poisoned it",
          len(measured) == 12 and all(np.all(np.isfinite(m)) for m in measured), len(measured))
    check("the bad row keeps its finite executed action and names the bad force",
          np.all(np.isfinite(rows[11]["executed_action"]))
          and np.isnan(rows[11]["force18"][CHANNELS[0]]), None)
    check("one state row is still written for the last clean frame only",
          report["state_rows"] == len(rows), report)


def t_nonfinite_eef_rule():
    env = DaggerFixtureEnv(nan_eef_frame=10)
    ctrl = Controller(rules=[NonfiniteState("observation.eef_pose")], sticky="end")
    report, table, _m, _e, _p, expert, _c, _s = run(env=env, controller=ctrl)
    rows = table.rows
    check("a diverged pose hands the wheel over on the frame it is seen",
          rows[9]["source"] == "policy" and rows[10]["source"] == "expert", None)
    check("with the field named in the reason",
          rows[10]["reason"] == "nonfinite_state:observation.eef_pose", rows[10]["reason"])
    check("and the recovery runs to the end of the episode",
          report["by_source"]["expert"] == N_STEPS - 10, report["by_source"])
    check("the bad pose itself is stored, on the row that visited it",
          rows[-1]["frame"] == N_STEPS - 1 and np.any(np.isnan(rows[9]["eef_pose"])),
          rows[9]["eef_pose"].tolist())


def t_peek_refused_for_stateful_expert():
    expert = StatefulExpert()
    report, table, _m, _e, _p, _x, _c, _s = run(expert=expert, peek_expert=True)
    rows = table.rows
    check("a stateful expert is never peeked: it is not advanced by a policy-controlled frame",
          expert.act_calls == 0, expert.act_calls)
    check("and the rows say so instead of storing a borrowed action",
          all(np.all(np.isnan(r["expert_action"])) for r in rows)
          and all(r["note"] == "stateful_expert_peek_refused" for r in rows), rows[0]["note"])
    check("label_kind stays none, so no counterfactual supervision is claimed",
          all(r["label_kind"] == "none" for r in rows), None)
    check("the refusal is counted in the report, not only written in the rows",
          report["by_note"].get("stateful_expert_peek_refused") == N_STEPS, report["by_note"])


def t_counterfactual_labels_from_stateless_expert():
    expert = StatelessExpert()
    report, table, _m, _e, _p, _x, _c, _s = run(expert=expert, peek_expert=True)
    rows = table.rows
    check("a stateless expert can be queried on every visited frame",
          all(np.all(np.isfinite(r["expert_action"])) for r in rows), None)
    check("those rows are labelled counterfactual, never executed",
          all(r["label_kind"] == "counterfactual" for r in rows)
          and report["by_label_kind"]["counterfactual"] == N_STEPS, report["by_label_kind"])
    check("the counterfactual action is a function of the visited observation only",
          np.allclose(rows[6]["expert_action"], 0.5 * 6 * 0.01), rows[6]["expert_action"][0])
    check("and it is NOT counted as a trainable label -- a takeover is still required",
          report["trainable_label_rows"] == 0, report)
    try:
        labels_for_training(table)
        ok = False
    except DaggerIntegrityError:
        ok = True
    check("exporting labels from an expert-free episode raises rather than falling back", ok)


def t_labels_are_only_genuine_takeovers():
    ctrl = Controller(rules=[ManualFrames([(7, 8)])], sticky="end")
    _report, table, _m, _e, _p, _x, _c, _s = run(controller=ctrl, expert=StatelessExpert(),
                                                  peek_expert=True)
    labels = labels_for_training(table)
    check("labels_for_training returns exactly the 16 takeover frames",
          [r["frame"] for r in labels] == list(range(8, N_STEPS)), len(labels))
    check("every returned label has a finite expert action equal to what was executed",
          all(np.array_equal(r["expert_action"], r["executed_action"])
              and np.all(np.isfinite(r["expert_action"])) for r in labels), None)
    check("and no policy-controlled frame leaks in",
          not any(r["source"] == "policy" for r in labels), None)


def t_verify_rows_catches_lies():
    good = StepTable(action_dim=ACTION_DIM, force_dim=FORCE_DIM)
    a = np.ones(ACTION_DIM)
    base = dict(frame=0, seed=1, episode_id=0, applied_action=a, executed_action=a,
                ts_wall=1.0, ts_mono=1.0, force18=np.zeros(FORCE_DIM),
                force6=np.zeros(6), pred_force=None, reward=0.0,
                terminated=False, truncated=False, note="")
    good.add(policy_action=a, expert_action=np.full(ACTION_DIM, np.nan), source="policy",
             label_kind="none", intervention=0, reason="policy",
             eef_pose=np.zeros(7), opening=0.0, cookies_in_target=0, **base)
    check("a truthful table verifies with no problems", verify_rows(good.rows) == [], good.rows[0])

    bad = StepTable(action_dim=ACTION_DIM, force_dim=FORCE_DIM)
    bad.add(policy_action=np.full(ACTION_DIM, np.nan), expert_action=np.full(ACTION_DIM, np.nan),
            source="policy", label_kind="executed", intervention=0, reason="policy",
            eef_pose=np.zeros(7), opening=0.0, cookies_in_target=0, **base)
    problems = verify_rows(bad.rows)
    check("a policy row without a policy action is caught",
          any("no policy action" in p for p in problems), problems)
    check("an expert label claimed on a policy-controlled frame is caught",
          any("expert label claimed" in p for p in problems), problems)

    swapped = StepTable(action_dim=ACTION_DIM, force_dim=FORCE_DIM)
    swapped.add(policy_action=a, expert_action=2 * a, source="expert", label_kind="executed",
                intervention=1, reason="manual:f0", eef_pose=np.zeros(7), opening=0.0,
                cookies_in_target=0, ts_wall=2.0, ts_mono=2.0, frame=0, seed=1, episode_id=0,
                applied_action=a, executed_action=a, force18=np.zeros(FORCE_DIM),
                force6=np.zeros(6), pred_force=None, reward=0.0, terminated=False,
                truncated=False, note="")
    problems = verify_rows(swapped.rows)
    check("an expert row whose executed action is not the expert's is caught",
          any("executed != expert action" in p for p in problems), problems)

    unflagged = StepTable(action_dim=ACTION_DIM, force_dim=FORCE_DIM)
    unflagged.add(policy_action=np.full(ACTION_DIM, np.nan), expert_action=a, source="expert",
                  label_kind="executed", intervention=0, reason="policy", eef_pose=np.zeros(7),
                  opening=0.0, cookies_in_target=0, ts_wall=3.0, ts_mono=3.0, frame=0, seed=1,
                  episode_id=0, applied_action=a, executed_action=a,
                  force18=np.zeros(FORCE_DIM), force6=np.zeros(6), pred_force=None, reward=0.0,
                  terminated=False, truncated=False, note="")
    problems = verify_rows(unflagged.rows)
    check("an expert-controlled frame that forgot to set the intervention flag is caught",
          any("intervention flag disagrees" in p for p in problems), problems)

    clock = StepTable(action_dim=ACTION_DIM, force_dim=FORCE_DIM)
    clock.add(policy_action=a, expert_action=np.full(ACTION_DIM, np.nan), source="policy",
              label_kind="none", intervention=0, reason="policy", eef_pose=np.zeros(7),
              opening=0.0, cookies_in_target=0, frame=1, seed=1, episode_id=0,
              applied_action=a, executed_action=a, force18=np.zeros(FORCE_DIM),
              force6=np.zeros(6), pred_force=None, reward=0.0, terminated=False,
              truncated=False, ts_wall=0.5, ts_mono=0.5, note="")
    clock.add(policy_action=a, expert_action=np.full(ACTION_DIM, np.nan), source="policy",
              label_kind="none", intervention=0, reason="policy", eef_pose=np.zeros(7),
              opening=0.0, cookies_in_target=0, frame=2, seed=1, episode_id=0,
              applied_action=a, executed_action=a, force18=np.zeros(FORCE_DIM),
              force6=np.zeros(6), pred_force=None, reward=0.0, terminated=False,
              truncated=False, ts_wall=0.4, ts_mono=0.4, note="")
    problems = verify_rows(clock.rows)
    check("a clock that goes backwards between frames is caught",
          any("wall clock went backwards" in p for p in problems)
          and any("monotonic clock went backwards" in p for p in problems), problems)

    mixed = StepTable(action_dim=ACTION_DIM, force_dim=FORCE_DIM)
    mixed.add(policy_action=a, expert_action=np.full(ACTION_DIM, np.nan), source="policy",
              label_kind="none", intervention=0, reason="policy", eef_pose=np.zeros(7),
              opening=0.0, cookies_in_target=0, frame=0, seed=1, episode_id=0,
              applied_action=a, executed_action=a, force18=np.zeros(FORCE_DIM),
              force6=np.zeros(6), pred_force=None, reward=0.0, terminated=False,
              truncated=False, ts_wall=1_782_703_536.79, ts_mono=-1_782_703_536.79, note="")
    problems = verify_rows(mixed.rows)
    check("an epoch-scale ts_mono -- the two clocks mixed -- is caught",
          any("ts_mono is not an elapsed time" in p for p in problems), problems)


def t_timestamps_and_frames():
    _report, table, _m, _e, _p, _x, _c, _s = run()
    rows = table.rows
    check("the frame column is exactly 0..steps-1",
          [r["frame"] for r in rows] == list(range(N_STEPS)), None)
    check("wall-clock timestamps are non-decreasing and real epoch seconds",
          all(rows[i + 1]["ts_wall"] >= rows[i]["ts_wall"] for i in range(len(rows) - 1))
          and rows[0]["ts_wall"] > 1_700_000_000, rows[0]["ts_wall"])
    check("monotonic offsets start near zero and increase",
          0.0 <= rows[0]["ts_mono"] < 5.0
          and all(rows[i + 1]["ts_mono"] > rows[i]["ts_mono"] for i in range(len(rows) - 1)),
          rows[0]["ts_mono"])
    check("the two clocks are not mixed: ts_mono is elapsed, ts_wall is epoch",
          all(abs(r["ts_mono"]) < 3600.0 for r in rows)
          and all(r["ts_wall"] > 1_700_000_000 for r in rows),
          {"ts_mono_first": rows[0]["ts_mono"], "ts_wall_first": rows[0]["ts_wall"]})
    check("the report carries first and last timestamps so a reader can audit the wall",
          _report["ts_first"] is not None and _report["ts_last"] >= _report["ts_first"], _report)


def t_npz_round_trip():
    ctrl = Controller(rules=[ManualFrames([(7, 8)])], sticky="end")
    with tempfile.TemporaryDirectory() as td:
        _report, _table, _m, _e, _p, _x, _c, _s = run(controller=ctrl, out_dir=td)
        p = Path(td) / "seed7_ep0" / "dagger.npz"
        check("the episode writes a dagger.npz and a report.json",
              p.exists() and (p.parent / "report.json").exists(), str(p.parent))
        z = np.load(p)
        check("npz row count matches the table", z["executed_action"].shape == (N_STEPS, ACTION_DIM),
              z["executed_action"].shape)
        check("the r3-compatible action14 column is the executed action, not the expert's intent",
              np.array_equal(z["action14"], z["executed_action"]), None)
        check("under takeover the executed action is the expert's and the policy column is NaN",
              np.all(np.isnan(z["policy_action"][8:]))
              and np.allclose(z["executed_action"][8], np.full(ACTION_DIM, 0.5)), None)
        check("source codes and names agree",
              list(z["source_name"]) == ["policy"] * 8 + ["expert"] * 16, None)
        check("state rows are one more than action rows and start at frame 0",
              z["state_force18"].shape[0] == N_STEPS + 1 and z["state_frame"][0] == 0
              and z["state_frame"][-1] == N_STEPS, z["state_frame"][-1])
        check("the terminal state is the last visited observation's force",
              np.allclose(z["state_force18"][-1], force18(N_STEPS)), None)
        check("timestamps survive the round trip",
              np.all(np.diff(z["ts_mono"]) > 0) and np.all(np.diff(z["ts_wall"]) >= 0), None)
        rep = json.loads((p.parent / "report.json").read_text())
        z.close()
        check("the report on disk says the same thing the table said",
              rep["rows"] == N_STEPS and rep["trainable_label_rows"] == 16
              and rep["integrity_ok"], rep["rows"])


def t_shadow_verdict_can_drive_a_stated_takeover():
    # the rule list is what makes the verdict a control decision; without it nothing happens
    fs = FakeShadow(verdicts=("stop",))
    _r0, table0, _m, _e, _p, _x, _c, s0 = run(shadow=FakeShadow(verdicts=("proceed",)))
    check("a shadow verdict alone never changes control: proceed -> 24 policy rows",
          all(r["source"] == "policy" for r in table0.rows), None)

    ctrl = Controller(rules=[ShadowVerdict(verdicts=("stop",))], sticky="end")
    report, table, _m, _e, policy, expert, _c, _s = run(controller=ctrl, shadow=fs)
    rows = table.rows
    check("with ShadowVerdict enabled a stop verdict hands over from the next frame",
          report["by_source"]["expert"] == N_STEPS - 1, report["by_source"])
    check("the reason names the verdict so the audit trail separates observation from action",
          rows[1]["reason"] == "shadow_verdict:stop", rows[1]["reason"])
    check("the monitor is fed one payload per policy decision, never per step",
          fs.calls == [0], fs.calls)
    check("and the model is not chunked again while the expert recovers",
          policy.chunk_calls == [0], policy.chunk_calls)

    from shadow import ShadowMonitor                       # the real monitor, numpy only
    calib = {"version": "force-calib-v1", "channels": list(CHANNELS),
             "mean": [0.0] * 6, "std": [1.0] * 6, "scale": 1.0}
    mon = ShadowMonitor(calib=calib, log_path=None, horizon=HORIZON, hist_len=HIST_LEN)
    ctrl2 = Controller(rules=[ShadowVerdict(verdicts=("hold", "stop"))], sticky="end")
    report2, _t2, _m2, _e2, _p2, _x2, _c2, _s2 = run(shadow=mon, controller=ctrl2)
    fin = mon.finalize()
    check("an off-band prediction is a hold verdict, which the stated rule turns into a takeover",
          fin["verdict_counts"].get("hold", 0) >= 1 and report2["intervention_rows"] > 0,
          {"verdicts": fin["verdict_counts"], "rows": report2["intervention_rows"]})
    check("the real shadow monitor still reports zero mutated actions under all of this",
          fin["actions_mutated"] == 0 and fin["mode"] == "shadow", fin["actions_mutated"])
    check("and its feedback-applied counter stays at zero",
          fin["feedback_applied_decisions"] == 0, fin["feedback_applied_decisions"])
    check("the takeover is accounted by the collector, not by the monitor",
          report2["intervention_rows"] == N_STEPS - 1, report2["intervention_rows"])


def t_mixed_episode_bookkeeping():
    ctrl = Controller(rules=[ManualFrames([(7, 6)])], sticky="end")
    report, table, _m, _e, _p, _x, _c, _s = run(controller=ctrl, expert=StatelessExpert(),
                                                peek_expert=True)
    rows = table.rows
    check("counterfactual rows before the takeover and executed rows after are both present",
          report_kind_counts(table) == {"counterfactual": 6, "executed": 18},
          report_kind_counts(table))
    check("a counterfactual row is never marked as an intervention",
          all((r["intervention"] == 0) == (r["source"] == "policy") for r in rows), None)
    check("the reason column for policy frames is exactly the string policy",
          all(r["reason"] == "policy" for r in rows if r["source"] == "policy"), None)


def report_kind_counts(table):
    out = {}
    for r in table.rows:
        out[r["label_kind"]] = out.get(r["label_kind"], 0) + 1
    return out


def t_describe_and_contract():
    from dagger5 import CONTRACT_VERSION
    report0, table0, *_rest = run()
    check("the report carries the collector contract version",
          report0["contract"] == CONTRACT_VERSION, report0["contract"])
    check("an empty rule set is reported as an empty rule set",
          report0["controller"]["rules"] == []
          and report0["controller"]["sticky"] == "end", report0["controller"])
    ctrl = Controller(rules=[ForceZAbove(CALIB_MEAN, CALIB_STD, z=4.0),
                             NoProgress(patience=50)], sticky="rule")
    report, table, *_r = run(controller=ctrl)
    rules = report["controller"]["rules"]
    check("two rules are reported in evaluation order with their thresholds",
          [x["rule"] for x in rules] == ["force_z", "no_progress"]
          and rules[0]["z"] == 4.0 and rules[1]["patience"] == 50, rules)
    check("rule order decides which reason a frame gets when several fire",
          all(r["reason"].startswith("force_z") or r["reason"].startswith("sticky")
              for r in table.rows if r["intervention"]),
          [r["reason"] for r in table.rows if r["intervention"]][:3])


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    return str(o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    for fn in [t_clean_episode, t_pred_alignment, t_manual_takeover_sticky_end,
               t_sticky_rule_and_release_after, t_force_z_rule, t_no_progress_rule,
               t_nan_force_break_path, t_nonfinite_eef_rule,
               t_peek_refused_for_stateful_expert,
               t_counterfactual_labels_from_stateless_expert, t_labels_are_only_genuine_takeovers,
               t_verify_rows_catches_lies, t_timestamps_and_frames, t_npz_round_trip,
               t_shadow_verdict_can_drive_a_stated_takeover, t_mixed_episode_bookkeeping,
               t_describe_and_contract]:
        try:
            fn()
        except Exception as exc:                                  # noqa: BLE001
            check(f"{fn.__name__} raised", False, f"{type(exc).__name__}: {exc}")
    RESULTS["all_passed"] = RESULTS["n_passed"] == RESULTS["n_checks"]
    print(json.dumps({k: RESULTS[k] for k in
                      ("n_checks", "n_passed", "all_passed", "failed")}, indent=2))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(RESULTS, indent=2, default=_jsonable))
        print(f"json -> {args.json}")
    return 0 if RESULTS["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
