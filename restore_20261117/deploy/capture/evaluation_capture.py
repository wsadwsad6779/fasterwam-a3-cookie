"""Wire already-loaded policy/env into an external run_episode callable, CPU-safe.

Call run_captured(run_episode, env=env, policy=policy, reset=reset, seed=seed,
                  artifacts={'weights': [...], 'config': [...], 'source': [...]},
                  max_steps=600, replan=10).
Returns (the unchanged run_episode result, Capture). No launcher is provided.
Not thread-safe: do not concurrently use the same policy instance during a chunk.
"""
from __future__ import annotations

from execution_chain_capture import Capture, copy_cpu


class _Proxy:
    def __init__(self, target):
        self._target = target

    def __getattr__(self, name):
        return getattr(self._target, name)


class _Bridge(_Proxy):
    def __init__(self, target):
        super().__init__(target)
        self.calls = []

    def denormalize_action(self, action, *args, **kwargs):
        raw = copy_cpu(action)
        result = self._target.denormalize_action(action, *args, **kwargs)
        self.calls.append((raw, copy_cpu(result)))
        return result


class _Adapter(_Proxy):
    def __init__(self, target):
        super().__init__(target)
        self.rows = []
        self.outputs = []

    def convert(self, action, *args, **kwargs):
        self.rows.append(copy_cpu(action))
        result = self._target.convert(action, *args, **kwargs)
        self.outputs.append(copy_cpu(result))
        return result


class CapturedPolicy(_Proxy):
    def __init__(self, policy, capture):
        super().__init__(policy)
        self.capture = capture

    def chunk(self, obs, frame, seed):
        if obs is not self.capture.observation or frame != self.capture.frame:
            raise ValueError("chunk must use current observation and frame")
        policy = self._target
        bridge, adapter = policy.bridge, policy.adapter
        # Restore exact instance attribute presence, including inherited attributes.
        saved = {name: (name in vars(policy), vars(policy).get(name))
                 for name in ("bridge", "adapter")}
        bridge_proxy, adapter_proxy = _Bridge(bridge), _Adapter(adapter)
        try:
            policy.bridge = bridge_proxy
            policy.adapter = adapter_proxy
            result = policy.chunk(obs, frame=frame, seed=seed)
            if len(bridge_proxy.calls) != 1:
                raise ValueError("expected exactly one denormalize_action per chunk")
            raw, denorm = bridge_proxy.calls[0]
            self.capture.chunk(frame, raw, denorm, adapter_proxy.rows, adapter_proxy.outputs)
            actions = result[0]
            import numpy as np
            recorded = self.capture.arrays[f"chunk_{len(self.capture.chunks)-1}_adapter"]
            if len(actions) != len(recorded) or any(
                not np.array_equal(a, b) for a, b in zip(actions, recorded)
            ):
                raise ValueError("returned actions differ from adapter outputs")
            return result
        finally:
            for name, (present, value) in saved.items():
                if present:
                    setattr(policy, name, value)
                elif name in vars(policy):
                    delattr(policy, name)


class CapturedEnv(_Proxy):
    def __init__(self, env, capture):
        super().__init__(env)
        self.capture = capture

    def step(self, action):
        handed = self.capture.selected(action)
        result = self._target.step(action)
        obs, _, _, _, info = result
        self.capture.executed(handed, obs, info)
        return result


def run_captured(run_episode, *, env, policy, reset, seed, artifacts, **kwargs):
    capture = Capture(artifacts)
    wrapped_env = CapturedEnv(env, capture)
    wrapped_policy = CapturedPolicy(policy, capture)

    def wrapped_reset(received_env, received_seed):
        if received_env is not wrapped_env:
            raise ValueError("reset received unexpected env")
        obs = reset(env, received_seed)
        capture.reset(obs)
        return obs

    try:
        capture.status = "running"
        result = run_episode(env=wrapped_env, policy=wrapped_policy, seed=seed,
                             reset=wrapped_reset, **kwargs)
        capture.status = "complete"
        return result, capture
    except BaseException as exc:
        capture.status = "failed"
        capture.error = f"{type(exc).__name__}: {exc}"
        # Expose partial evidence without replacing the original failure.
        try:
            exc.execution_capture = capture
        except (AttributeError, TypeError):
            pass
        raise
