"""P5 corrective-data collector: record what the policy wanted, what the expert did, and who was
actually holding the wheel -- one row per executed simulator step.

The r3-era collector (`/workspace/fasterwam_a3_bridge/bridge/_next/dagger_recover.py`) replayed a
recorded prefix, snapshotted the physics state, then ran the scripted expert to completion and
saved only the expert's actions. That is fine for "give me recovery demonstrations" and wrong for
DAgger: DAgger's whole premise is labelling the states *the student visits*, so a table that only
contains expert-owned frames cannot tell you where the policy was about to fail, and a table that
quietly pastes expert actions onto policy frames would be fabricated supervision.

What this module adds, per executed step:

    policy_action   the action the model's own chunk put on the wire at this frame (NaN if the
                    model was not in control, or was not asked because it is mid-takeover)
    expert_action   the scripted expert's action at this frame (NaN when the expert was not
                    queried, or when it was asked but the expert refuses counterfactual queries)
    applied_action  `info["applied_action"]`, what the simulator says it actually executed
    source          who was in control: policy / expert / none
    label_kind      executed (a real takeover) | counterfactual (expert spoke, policy acted) |
                    none (no label exists for this frame)
    intervention    1 when the frame was handed over by a rule, with the rule id and threshold
    ts              wall-clock epoch seconds and a monotonic clock, both per frame
    force           the raw 18-channel vector plus the six frozen channels

The three honesty gates that decide whether a row may be called a training label are enforced in
`verify_rows` and re-checked by `labels_for_training`, which refuses an empty or unverifiable
table instead of falling back to something convenient:

* a takeover is a takeover only if the executed action is the expert's action -- the shadow
  monitor's verdict alone never produces a label, because in shadow mode nothing is mutated;
* a stateful expert is never peeked. `A3SameColumnBatchExpert.act` advances its own phase
  machine, so calling it on frames that are then not executed corrupts the expert. Counterfactual
  labels are only collected from a provider that declares `stateless_peek = True`;
* the policy's remaining chunk actions are dropped, not deferred, when control leaves the model,
  and the count is reported. Executing stale model actions after a recovery would describe a
  policy nobody trained.

Nothing in this file imports torch or mujoco: it takes an env, a policy and an expert as duck-
typed arguments, so the CPU self-test in `dagger_selftest.py` runs the same code path the real
collector runs.

    cd /workspace/_v1_stage/dev/p5_dagger
    PYTHONDONTWRITEBYTECODE=1 python dagger_selftest.py
    PYTHONDONTWRITEBYTECODE=1 python dagger_collect.py --help        # real env, needs the GPU node
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "eval40k"))

from force_stream import FORCE_OBS_KEY, select_channels          # noqa: E402

NAN = float("nan")
SOURCE_CODES = {"policy": 0, "expert": 1, "none": 2}
SOURCES = tuple(SOURCE_CODES)
LABEL_KINDS = ("executed", "counterfactual", "none")
CONTRACT_VERSION = "p5-dagger-v1"


class DaggerIntegrityError(RuntimeError):
    """The recorded table does not describe what the loop claims to have done."""


# --------------------------------------------------------------------------- rows

def _nan_row(n):
    return np.full(n, NAN, dtype=np.float64)


class StepTable:
    """One row per executed simulator step, held column-wise until the episode is written."""

    def __init__(self, action_dim=14, force_dim=18, frozen_dim=6):
        self.action_dim = int(action_dim)
        self.force_dim = int(force_dim)
        self.frozen_dim = int(frozen_dim)
        self.rows = []
        self.states = []

    def add_state(self, *, frame, force18, eef_pose, opening, cookies_in_target):
        """Record a visited state, one more than the number of actions.

        `prepare_a3_arrays.py` labels `action[t]` with the delta to state `t+1`, so a converter
        needs the terminal state as well as the state each action was taken from; the r3
        collector kept `state_rows == action_rows + 1` and this keeps the same invariant.
        """
        self.states.append({"frame": int(frame),
                            "force18": np.asarray(force18, np.float64),
                            "eef_pose": np.asarray(eef_pose, np.float64),
                            "opening": float(opening),
                            "cookies_in_target": int(cookies_in_target)})

    def n_states(self):
        return len(self.states)

    def add(self, *, frame, seed, episode_id, source, policy_action, expert_action,
            applied_action, executed_action, label_kind, intervention, reason, ts_wall,
            ts_mono, force18, force6, pred_force, eef_pose, opening, cookies_in_target,
            reward, terminated, truncated, note=None):
        self.rows.append({
            "frame": int(frame), "seed": int(seed), "episode_id": int(episode_id),
            "source": str(source), "policy_action": np.asarray(policy_action, np.float64),
            "expert_action": np.asarray(expert_action, np.float64),
            "applied_action": np.asarray(applied_action, np.float64),
            "executed_action": np.asarray(executed_action, np.float64),
            "label_kind": str(label_kind), "intervention": int(intervention),
            "reason": str(reason), "ts_wall": float(ts_wall), "ts_mono": float(ts_mono),
            "force18": np.asarray(force18, np.float64),
            "force6": np.asarray(force6, np.float64),
            "pred_force": None if pred_force is None else np.asarray(pred_force, np.float64),
            "eef_pose": np.asarray(eef_pose, np.float64), "opening": float(opening),
            "cookies_in_target": int(cookies_in_target), "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "note": "" if note is None else str(note),
        })

    def __len__(self):
        return len(self.rows)

    def arrays(self):
        n = len(self.rows)
        if n == 0:
            return {}

        def col(key, shape, dtype=np.float64):
            out = np.zeros((n,) + tuple(shape), dtype=dtype)
            for i, r in enumerate(self.rows):
                out[i] = r[key]
            return out

        arr = {
            "frame": np.array([r["frame"] for r in self.rows], np.int64),
            "seed": np.array([r["seed"] for r in self.rows], np.int64),
            "episode_id": np.array([r["episode_id"] for r in self.rows], np.int64),
            "source": np.array([SOURCE_CODES[r["source"]] for r in self.rows], np.int8),
            "source_name": np.array([r["source"] for r in self.rows], dtype="U8"),
            "policy_action": col("policy_action", (self.action_dim,)),
            "expert_action": col("expert_action", (self.action_dim,)),
            "applied_action": col("applied_action", (self.action_dim,)),
            "executed_action": col("executed_action", (self.action_dim,)),
            "label_kind": np.array([r["label_kind"] for r in self.rows], dtype="U14"),
            "intervention": np.array([r["intervention"] for r in self.rows], np.int8),
            "reason": np.array([r["reason"] for r in self.rows], dtype="U32"),
            "ts_wall": np.array([r["ts_wall"] for r in self.rows], np.float64),
            "ts_mono": np.array([r["ts_mono"] for r in self.rows], np.float64),
            "force18": col("force18", (self.force_dim,)),
            "force6": col("force6", (self.frozen_dim,)),
            "eef_pose": col("eef_pose", (7,)),
            "opening": np.array([r["opening"] for r in self.rows], np.float64),
            "cookies_in_target": np.array([r["cookies_in_target"] for r in self.rows], np.int32),
            "reward": np.array([r["reward"] for r in self.rows], np.float64),
            "done": np.array([r["terminated"] or r["truncated"] for r in self.rows], bool),
            "note": np.array([r["note"] for r in self.rows], dtype="U48"),
        }
        # `action14` keeps the r3 name so the existing converter still finds the column: it is the
        # action that was actually executed, which is the only action a dataset may imitate.
        arr["action14"] = arr["executed_action"]
        if self.states:
            def scol(key, shape):
                out = np.zeros((len(self.states),) + tuple(shape), dtype=np.float64)
                for i, r in enumerate(self.states):
                    out[i] = r[key]
                return out
            arr["state_frame"] = np.array([r["frame"] for r in self.states], np.int64)
            arr["state_force18"] = scol("force18", (self.force_dim,))
            arr["state_eef_pose"] = scol("eef_pose", (7,))
            arr["state_opening"] = np.array([r["opening"] for r in self.states], np.float64)
            arr["state_cookies_in_target"] = np.array([r["cookies_in_target"] for r in self.states],
                                                      np.int32)
        return arr

    def write(self, path, extra=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.arrays())
        return {"npz": str(path), "rows": len(self.rows)}


# --------------------------------------------------------------------------- takeover rules

class Rule:
    """A named, threshold-bearing reason to hand the wheel to the expert.

    `check` is called with the frame, the observation before the step, the last `info`, the force
    rows measured so far and the newest shadow verdict. It returns a short reason string or None.
    Rules never see or touch an action: deciding to intervene is separate from intervening.
    """

    id = "rule"

    def __init__(self, **params):
        self.params = params

    def check(self, frame, obs, info, measured, shadow_verdict):
        raise NotImplementedError

    def describe(self):
        return {"rule": self.id, **self.params}


class NonfiniteState(Rule):
    id = "nonfinite_state"

    def __init__(self, field="observation.eef_pose"):
        super().__init__(field=field)
        self.field = field

    def check(self, frame, obs, info, measured, shadow_verdict):
        v = np.asarray(obs.get(self.field), dtype=np.float64)
        if v.size and not np.all(np.isfinite(v)):
            return f"{self.id}:{self.field}"
        return None


class ForceZAbove(Rule):
    """Any frozen channel beyond `z` calibration sigmas -- the same band the shadow monitor uses,
    but here the decision changes who acts, so it is written down as its own rule with its own
    threshold and gets audited on its own."""

    id = "force_z"

    def __init__(self, mean, std, z=6.0):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        if self.std.shape != self.mean.shape:
            raise DaggerIntegrityError("force_z rule: mean/std shape mismatch")
        if not np.all(self.std > 0):
            raise DaggerIntegrityError("force_z rule: std must be strictly positive")
        self.z = float(z)
        super().__init__(z=self.z, channels=int(self.mean.size),
                         mean=self.mean.tolist(), std=self.std.tolist())

    def check(self, frame, obs, info, measured, shadow_verdict):
        if not measured:
            return None
        z = np.abs(np.asarray(measured[-1], dtype=np.float64) - self.mean) / self.std
        bad = np.flatnonzero(z > self.z)
        if bad.size:
            return f"{self.id}:c{int(bad[0])}=z{float(z[bad[0]]):.1f}"
        return None


class NoProgress(Rule):
    """Intervene when the task counter has not moved for `patience` frames since the last gain."""

    id = "no_progress"

    def __init__(self, patience=120, key="cookies_in_target"):
        self.patience = int(patience)
        self.key = key
        self._best = None
        self._since = 0
        super().__init__(patience=self.patience, key=key)

    def check(self, frame, obs, info, measured, shadow_verdict):
        value = int((info or {}).get(self.key, -1))
        if self._best is None or value > self._best:
            self._best = value
            self._since = int(frame)
            return None
        if int(frame) - self._since >= self.patience:
            return f"{self.id}:{self.key}={value}x{int(frame) - self._since}f"
        return None


class ShadowVerdict(Rule):
    """Take over when the P4 shadow monitor says `stop` (or `hold`).

    This is the one place a shadow verdict is allowed to change behaviour, and it only happens
    because the collector explicitly asks for it: the monitor itself never returns an action, and
    `ShadowMonitor.finalize` still reports `actions_mutated = 0` for the monitor's own accounting.
    The reason string names the verdict so the audit trail separates the two.
    """

    id = "shadow_verdict"

    def __init__(self, verdicts=("stop",)):
        self.verdicts = set(verdicts)
        super().__init__(verdicts=sorted(self.verdicts))

    def check(self, frame, obs, info, measured, shadow_verdict):
        if shadow_verdict in self.verdicts:
            return f"{self.id}:{shadow_verdict}"
        return None


class ManualFrames(Rule):
    """Explicit (seed, frame) takeover points, the direct successor of `--take 90`."""

    id = "manual"

    def __init__(self, points=(), seed=None):
        self.points = {(int(s), int(f)) for s, f in points}
        self.seed = None if seed is None else int(seed)
        super().__init__(n_points=len(self.points), seed=self.seed)

    def check(self, frame, obs, info, measured, shadow_verdict):
        f = int(frame)
        if self.seed is None:
            hit = any(int(p[1]) == f for p in self.points)
        else:
            hit = (self.seed, f) in self.points
        return f"{self.id}:f{f}" if hit else None


class Controller:
    """Policy-vs-expert switch with a stated, auditable rule set.

    `sticky="end"` keeps the expert in control to the end of the episode, which is what the r3
    collector did and what makes the recovered prefix imitable; `sticky="rule"` gives control
    back as soon as no rule fires, which is the textbook DAgger blend and produces both label
    kinds in one episode.
    """

    def __init__(self, rules=(), sticky="end", release_after=None):
        self.rules = list(rules)
        self.sticky = str(sticky)
        self.release_after = None if release_after is None else int(release_after)
        self._in_handover = False
        self._since = None
        self.decisions = []

    def describe(self):
        return {"sticky": self.sticky, "release_after": self.release_after,
                "rules": [r.describe() for r in self.rules]}

    def decide(self, frame, obs, info, measured, shadow_verdict=None):
        """Return `(takeover: bool, reason: str)` for the frame about to be acted on."""
        fired = None
        for rule in self.rules:
            got = rule.check(frame, obs, info, measured, shadow_verdict)
            if got:
                fired = got
                break
        if fired:
            self._in_handover = True
            self._since = self._since if self._since is not None else int(frame)
        elif self._in_handover:
            if self.sticky == "end":
                fired = "sticky:end"
            elif self.release_after is not None and int(frame) - self._since < self.release_after:
                fired = f"sticky:{int(frame) - self._since}f_of{self.release_after}"
            else:
                self._in_handover = False
        takeover = bool(fired)
        self.decisions.append({"frame": int(frame), "takeover": takeover,
                               "reason": fired or "policy"})
        return takeover, (fired or "policy")


# --------------------------------------------------------------------------- expert probes

def expert_action(expert, obs, dim, allow_peek, in_control):
    """Ask the expert, or refuse, and always say which happened.

    Returns `(action_array, label_kind, note)`. A stateful expert queried on a frame that the
    policy then acts on would have its phase machine advanced by an action nobody executed, so
    the only honest answer there is NaN + a refusal note, unless the provider opts in with
    `stateless_peek = True`.
    """
    if in_control:
        a = np.asarray(expert.act(obs), dtype=np.float64).copy()
        if a.shape != (dim,):
            raise DaggerIntegrityError(f"expert returned {a.shape}, expected ({dim},)")
        if not np.all(np.isfinite(a)):
            raise DaggerIntegrityError("expert returned a non-finite action; refusing to record it")
        return a, "executed", ""
    if not allow_peek:
        return _nan_row(dim), "none", "peek_disabled"
    if not bool(getattr(expert, "stateless_peek", False)):
        return _nan_row(dim), "none", "stateful_expert_peek_refused"
    a = np.asarray(expert.act(obs), dtype=np.float64).copy()
    if a.shape != (dim,):
        raise DaggerIntegrityError(f"expert peek returned {a.shape}, expected ({dim},)")
    return a, "counterfactual", "stateless_peek"


# --------------------------------------------------------------------------- the loop

def _state_fields(obs, info, force_key):
    f18 = np.asarray(obs[force_key], dtype=np.float64)
    state = np.asarray(obs.get("observation.state", [])).reshape(-1)
    return {"force18": f18,
            "eef_pose": np.asarray(obs.get("observation.eef_pose", _nan_row(7)), np.float64),
            "opening": float(state[7]) if state.size > 7 else NAN,
            "cookies_in_target": int((info or {}).get("cookies_in_target", -1))}


def run_dagger_episode(env, policy, expert, reset, seed, *, out_dir=None, max_steps=600,
                       replan=10, controller=None, shadow=None, peek_expert=False,
                       episode_id=0, action_dim=14, force_key=FORCE_OBS_KEY,
                       force_dim=18, log_every=0, extra_meta=None):
    """Roll one corrective-data episode and return `(report, table, measured)`.

    Frame numbering is identical to `rollout40k.run_episode`: `reset` gives frame 0, after `k`
    `env.step` calls the loop looks at frame `k`, and `policy.observe(obs, frame)` runs on every
    executed step whether or not the model is in control -- the model's force history is 16
    consecutive 20 Hz frames, so skipping the push during a takeover would poison the history it
    sees when control comes back.
    """
    controller = controller or Controller(rules=())
    obs = reset(env, seed)
    policy.start_episode(obs, seed)
    if hasattr(expert, "reset"):
        expert.reset()
    first18 = np.asarray(obs[force_key], dtype=np.float64)
    measured = [select_channels(first18, where="frame 0")]
    raw18 = [first18.copy()]
    policy.observe(obs, 0)

    table = StepTable(action_dim=action_dim, force_dim=force_dim)
    table.add_state(frame=0, **_state_fields(obs, None, force_key))
    pending = []
    chunks = []
    discarded = 0
    last_pred = None
    info = {}
    reward = 0.0
    terminated = truncated = False
    nan_abort = -1
    t0 = time.time()
    m0 = time.monotonic()          # `ts_mono` is seconds since start; `ts_wall` is absolute

    for frame in range(max_steps):
        shadow_verdict = chunks[-1].get("shadow_verdict") if chunks else None
        takeover, reason = controller.decide(frame, obs, info, measured, shadow_verdict)
        source = "expert" if takeover else "policy"

        if takeover:
            if pending:
                discarded += len(pending)
                pending = []
            executed, label_kind, note = expert_action(
                expert, obs, action_dim, allow_peek=True, in_control=True)
            expert_row, policy_action_row, pred_row = executed, _nan_row(action_dim), None
        else:
            if not pending:
                actions, pred_force, meta = policy.chunk(obs, frame=frame, seed=seed)
                take = [np.asarray(a, dtype=np.float64) for a in actions[:replan]]
                pf = np.asarray(pred_force, dtype=np.float64)
                last_pred = pf
                entry = {"frame": int(frame), "pred_force": pf, "n_actions": len(take),
                         "meta": dict(meta or {})}
                if shadow is not None:
                    doc = shadow.on_decision({"frame": int(frame), "seed": int(seed),
                                              "pred_force": pf, "measured": measured,
                                              "meta": meta or {}})
                    entry["shadow_verdict"] = doc.get("verdict")
                chunks.append(entry)
                pending = [(a, k) for k, a in enumerate(take)]
            executed, step_in_chunk = pending.pop(0)
            policy_action_row = executed
            expert_row, label_kind, note = expert_action(
                expert, obs, action_dim, allow_peek=peek_expert, in_control=False)
            # pred_force[j] forecasts the frame issued_at + 1 + j, and this row's step is
            # step_in_chunk after the issue frame, so the same index lines up with the next frame
            pred_row = None if last_pred is None \
                else last_pred[min(step_in_chunk, len(last_pred) - 1)]

        t_step = time.time()
        obs, reward, terminated, truncated, info = env.step(executed)
        applied = np.asarray(info.get("applied_action", _nan_row(action_dim)), dtype=np.float64)
        next_frame = frame + 1
        f18 = np.asarray(obs[force_key], dtype=np.float64)
        if not np.all(np.isfinite(f18)):
            # keep the frame that broke: a diverged force is the most informative row a
            # corrective collector can have, and `select_channels` refuses to normalise it
            fields = _state_fields(obs, info, force_key)
            table.add(frame=frame, seed=seed, episode_id=episode_id, source=source,
                      policy_action=policy_action_row, expert_action=expert_row,
                      applied_action=applied, executed_action=executed, label_kind=label_kind,
                      intervention=1 if takeover else 0, reason=reason, ts_wall=t_step,
                      ts_mono=time.monotonic() - m0, force18=f18,
                      force6=np.full(len(measured[-1]), NAN), pred_force=pred_row,
                      eef_pose=fields["eef_pose"], opening=fields["opening"],
                      cookies_in_target=fields["cookies_in_target"], reward=float(reward),
                      terminated=terminated, truncated=truncated,
                      note="nonfinite_force_after_step")
            nan_abort = next_frame
            break

        fields = _state_fields(obs, info, force_key)
        table.add(frame=frame, seed=seed, episode_id=episode_id, source=source,
                  policy_action=policy_action_row, expert_action=expert_row,
                  applied_action=applied, executed_action=executed, label_kind=label_kind,
                  intervention=1 if takeover else 0, reason=reason, ts_wall=t_step,
                  ts_mono=time.monotonic() - m0, force18=raw18[-1], force6=measured[-1],
                  pred_force=pred_row, eef_pose=fields["eef_pose"], opening=fields["opening"],
                  cookies_in_target=fields["cookies_in_target"], reward=float(reward),
                  terminated=terminated, truncated=truncated, note=note)
        table.add_state(frame=next_frame, **fields)

        measured.append(select_channels(f18, where=f"frame {next_frame}"))
        raw18.append(f18.copy())
        policy.observe(obs, next_frame)
        if log_every and next_frame % log_every == 0:
            print(f"[dagger] seed={seed} frame={next_frame} source={source} "
                  f"reason={reason} in_target={info.get('cookies_in_target')}", flush=True)
        if terminated or truncated:
            break

    if shadow is not None and hasattr(shadow, "flush"):
        shadow.flush(seed, measured)

    problems = verify_rows(table.rows)
    report = summarize(table, chunks, controller, seed=seed, episode_id=episode_id,
                       discarded=discarded, nan_abort=nan_abort, wall_s=time.time() - t0,
                       problems=problems, extra_meta=extra_meta)
    if out_dir is not None:
        table.write(Path(out_dir) / f"seed{seed}_ep{episode_id}" / "dagger.npz")
        (Path(out_dir) / f"seed{seed}_ep{episode_id}" / "report.json").write_text(
            json.dumps(report, indent=2))
    return report, table, measured


# --------------------------------------------------------------------------- verification

def verify_rows(rows):
    """Re-derive the claims of the table from the rows themselves.

    Every problem this finds means the collector lied about a frame, so the caller must treat a
    non-empty list as an aborted episode, not a warning.
    """
    problems = []
    prev_mono = -np.inf
    prev_wall = -np.inf
    for i, r in enumerate(rows):
        tag = f"row{i}(frame={r['frame']})"
        if r["source"] not in SOURCES:
            problems.append(f"{tag}: unknown source {r['source']!r}")
        if r["label_kind"] not in LABEL_KINDS:
            problems.append(f"{tag}: unknown label_kind {r['label_kind']!r}")
        if r["ts_mono"] < prev_mono:
            problems.append(f"{tag}: monotonic clock went backwards")
        if r["ts_wall"] < prev_wall:
            problems.append(f"{tag}: wall clock went backwards")
        # `ts_mono` is seconds since the episode started, so an epoch-scale value can only mean the
        # two clocks were mixed -- and a mixed clock silently breaks every duration a reader derives
        if abs(r["ts_mono"]) > 1e6:
            problems.append(f"{tag}: ts_mono is not an elapsed time ({r['ts_mono']})")
        prev_mono, prev_wall = r["ts_mono"], r["ts_wall"]

        ex = r["executed_action"]
        if r["source"] == "policy":
            if not np.array_equal(ex, r["policy_action"]):
                problems.append(f"{tag}: executed != policy action under policy control")
            if np.any(np.isnan(r["policy_action"])):
                problems.append(f"{tag}: policy-controlled row with no policy action")
            if r["label_kind"] == "executed":
                problems.append(f"{tag}: expert label claimed on a policy-controlled frame")
        elif r["source"] == "expert":
            if not np.array_equal(ex, r["expert_action"]):
                problems.append(f"{tag}: executed != expert action under expert control")
            if r["label_kind"] != "executed":
                problems.append(f"{tag}: expert-controlled row not labelled as an executed takeover")
            if np.any(np.isnan(r["expert_action"])):
                problems.append(f"{tag}: expert-controlled row with no expert action")
        if bool(r["intervention"]) != (r["source"] == "expert"):
            problems.append(f"{tag}: intervention flag disagrees with the control source")
        if r["label_kind"] == "counterfactual" and r["source"] != "policy":
            problems.append(f"{tag}: counterfactual label recorded while not under policy control")
        if not np.all(np.isfinite(ex)):
            problems.append(f"{tag}: non-finite executed action")
    return problems


def labels_for_training(table, require_takeover=True):
    """The rows a P5 fine-tune may imitate: genuine takeovers, verified, nothing else.

    Counterfactual expert rows are deliberately *not* included here. They are useful for
    diagnostics (what the expert would have done), but including them in a behavioural-cloning
    loss means training on actions the system never executed from states it did reach -- a
    mixture that is only defensible with an explicit weighting decision, which is not this
    function's job. `table.rows` keeps both so that decision can be made later with the data
    already on disk.
    """
    rows = [r for r in table.rows if r["source"] == "expert" and r["label_kind"] == "executed"]
    if require_takeover and not rows:
        raise DaggerIntegrityError(
            "no genuine takeover rows: this episode has no corrective supervision to export, and "
            "writing policy frames as expert labels is exactly the fabrication this collector is "
            "built to prevent")
    return rows


def summarize(table, chunks, controller, *, seed, episode_id, discarded, nan_abort, wall_s,
              problems, extra_meta=None):
    rows = table.rows
    n = len(rows)
    if table.n_states() not in (0, n + 1):
        problems = problems + [f"state rows {table.n_states()} != action rows {n} + 1"]
    by_source = {s: sum(1 for r in rows if r["source"] == s) for s in SOURCES}
    by_reason = {}
    for r in rows:
        by_reason[r["reason"]] = by_reason.get(r["reason"], 0) + 1
    by_kind = {k: sum(1 for r in rows if r["label_kind"] == k) for k in LABEL_KINDS}
    by_note = {}
    for r in rows:
        if r["note"]:
            by_note[r["note"]] = by_note.get(r["note"], 0) + 1
    out = {
        "contract": CONTRACT_VERSION,
        "seed": int(seed),
        "episode_id": int(episode_id),
        "rows": n,
        "frames": n + 1,
        "state_rows": table.n_states(),
        "by_source": by_source,
        "by_label_kind": by_kind,
        "by_intervention_reason": by_reason,
        "by_note": by_note,
        "intervention_rows": sum(1 for r in rows if r["intervention"]),
        "takeover_episodes": int(any(r["intervention"] for r in rows)),
        "trainable_label_rows": by_kind["executed"],
        "discarded_policy_actions": int(discarded),
        "decision_chunks": len(chunks),
        "nan_abort_frame": int(nan_abort),
        "wall_s": round(wall_s, 2),
        "controller": controller.describe(),
        "integrity_problems": problems,
        "integrity_ok": not problems,
        "ts_first": rows[0]["ts_wall"] if n else None,
        "ts_last": rows[-1]["ts_wall"] if n else None,
    }
    if extra_meta:
        out.update(extra_meta)
    return out
