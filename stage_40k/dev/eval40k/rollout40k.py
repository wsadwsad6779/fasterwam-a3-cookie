"""One closed-loop episode, written so the same loop serves the real model and the flow test.

The loop is deliberately thin: reset, execute a replan-sized slice of the policy's action
chunk, hand every resulting observation back to the policy, stop on termination or on a
non-finite pose. It carries no task rules and no safety layer -- this is the bare-policy
(B0) condition the report quotes -- so the only thing that varies between checkpoints is
the weights.

The one rule worth stating out loud is `policy.observe(obs, frame)` on *every* executed
step, not only on the steps where a new chunk is requested. The model's force history is
16 consecutive 20 Hz frames (`force_windows` reads `s-15 .. s`); a stream advanced once per
replan would hand it 16 frames spaced `replan` apart, which is a different time base and a
distribution the model never saw during training. The assertion in `ForcePolicy.chunk`
turns that mistake into an error instead of a quietly worse success rate.

Frame numbering: the observation returned by `reset` is frame 0, and after `k` calls to
`env.step` the loop is looking at frame `k`. `measured[f]` is the six frozen force channels
at frame `f`, so a chunk issued at frame `t` predicts `measured[t+1 .. t+H]`.
"""
from __future__ import annotations

import time

import numpy as np

from force_stream import FORCE_OBS_KEY, select_channels


def _success_of(info):
    """The env's own verdict, plus how many cookies landed, taken from `info` only.

    No simulator truth is read here: `info` is what a deployment would have. The cookie
    positions live in `cookie_truth_log` and are recorded for diagnosis, never scored.
    """
    success = info.get("success")
    if success is None:
        success = bool((info.get("cookies_in_target") or 0) >= (info.get("cookies_to_place") or 1))
    return bool(success), info.get("cookies_in_target")


def run_episode(env, policy, seed, reset, max_steps=600, replan=10, shadow=None,
                log_every=0):
    """Roll one episode out. Returns `(record, measured, chunks)`.

    `reset(env, seed)` must return the initial observation; the caller owns env construction
    so scene randomisation stays in one place. `shadow` is an optional P4 observer: it gets
    a payload per decision and may only read -- it cannot return an action.
    """
    obs = reset(env, seed)
    policy.start_episode(obs, seed)
    measured = [select_channels(obs[FORCE_OBS_KEY], where="frame 0")]
    policy.observe(obs, 0)
    chunks = []
    pending = []
    t0 = time.time()
    nan_abort = -1
    step = -1
    info = {}
    for step in range(max_steps):
        if not pending:
            actions, pred_force, meta = policy.chunk(obs, frame=len(measured) - 1, seed=seed)
            take = [np.asarray(a, dtype=np.float64) for a in actions[:replan]]
            pending.extend(take)
            chunks.append({"frame": len(measured) - 1, "pred_force": pred_force,
                           "n_actions": len(take), "meta": meta})
            if shadow is not None:
                shadow.on_decision({"frame": len(measured) - 1, "seed": int(seed),
                                    "pred_force": pred_force, "measured": measured,
                                    "meta": meta})
        action = pending.pop(0)
        obs, _reward, terminated, truncated, info = env.step(action)
        frame = len(measured)
        f18 = np.asarray(obs[FORCE_OBS_KEY], dtype=np.float64)
        if not np.all(np.isfinite(f18)):
            nan_abort = frame
            break
        measured.append(select_channels(f18, where=f"frame {frame}"))
        policy.observe(obs, frame)
        if log_every and frame % log_every == 0:
            print(f"[episode seed={seed}] frame={frame} "
                  f"force={np.round(measured[-1], 2).tolist()} "
                  f"in_target={info.get('cookies_in_target')}", flush=True)
        if terminated or truncated:
            break

    success, in_target = _success_of(info)
    if shadow is not None and hasattr(shadow, "flush"):
        # the chunk issued just before termination has futures that may already have happened;
        # without this they would sit unscored and the coverage number would be a lie
        shadow.flush(seed, measured)
    record = {
        "seed": int(seed),
        "frames": len(measured),
        "steps_executed": len(measured) - 1,
        "decision_chunks": len(chunks),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": success,
        "cookies_in_target": None if in_target is None else int(in_target),
        "nan_abort_frame": int(nan_abort),
        "wall_s": round(time.time() - t0, 2),
    }
    return record, measured, chunks
