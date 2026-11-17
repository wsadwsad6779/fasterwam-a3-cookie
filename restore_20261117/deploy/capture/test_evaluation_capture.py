"""CPU-only contract acceptance; all weight/config artifacts here are synthetic fixtures."""
import ast
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import time
from types import SimpleNamespace
from pathlib import Path
import random
import sys
import unittest
import uuid

import numpy as np

from evaluation_capture import CapturedEnv, CapturedPolicy, run_captured
from execution_chain_capture import Capture

ROOT = Path(__file__).resolve().parent
ORIGINAL = Path('C:/Users/ASUS/Documents/Qoder/2026-10-31/b462d38d/_v1_stage/dev/eval40k')
sys.dont_write_bytecode = True


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


force = load_source('capture_test_force', ORIGINAL / 'force_stream.py')
previous = sys.modules.get('force_stream')
try:
    sys.modules['force_stream'] = force
    rollout = load_source('capture_test_rollout', ORIGINAL / 'rollout40k.py')
finally:
    if previous is None:
        del sys.modules['force_stream']
    else:
        sys.modules['force_stream'] = previous

ARTIFACTS = {'weights': [__file__], 'config': [__file__],
             'source': [__file__, str(ORIGINAL / 'rollout40k.py'),
                        str(ORIGINAL / 'policy40k.py'),
                        str(ORIGINAL / 'force_stream.py'),
                        str(ROOT / 'execution_chain_capture.py'),
                        str(ROOT / 'evaluation_capture.py')]}


class Bridge:
    def denormalize_action(self, action, processor):
        action += .125  # deliberately pollute raw, alias the returned denorm
        return action

    def quat_wxyz_to_matrix(self, quat):
        return np.eye(3)


class Adapter:
    def convert(self, action, rot):
        result = np.tile(action, 2)
        action[:] = 777  # deliberately pollute rewritten after input capture
        return result


SHARED_BRIDGE = Bridge()


class Policy:
    bridge = SHARED_BRIDGE
    adapter = Adapter()

    def __init__(self, rows=32, failure=None):
        self.rows, self.failure = rows, failure
        self.calls = 0
        self.observed = []
        self.results = []

    def start_episode(self, obs, seed):
        self.initial = obs

    def observe(self, obs, frame):
        self.observed.append((obs, frame))

    def chunk(self, obs, frame, seed):
        self.calls += 1
        raw = np.random.uniform(-.8, .8, (1, self.rows, 7))
        raw += random.random() * .01
        self.raw_before = raw.copy()
        action = self.bridge.denormalize_action(raw, None)[0]
        if self.failure == 'throw':
            raise RuntimeError('injected chunk failure')
        action[..., -1] = np.where(action[..., -1] >= 0, 1., -1.)
        rot = self.bridge.quat_wxyz_to_matrix(np.zeros(4))
        actions = [self.adapter.convert(a, rot) for a in action]
        if self.failure == 'bad_adapter':
            actions[0] = actions[0][:13]
        result = (actions, np.zeros((self.rows, 6)), {'replan_from_frame': frame})
        self.results.append(result)
        return result


class Env:
    def __init__(self, stop=100, failure=None):
        self.stop, self.failure = stop, failure
        self.calls = 0
        self.actions = []
        self.states = []
        self.results = []

    def obs(self):
        state = np.arange(16, dtype=float) * .01 + self.calls * .001
        state[7] = .2 + self.calls * .01
        obs = {'observation.state': state,
               'observation.eef_pose': np.array([0., 0., 0., 1., 0., 0., 0.]),
               'observation.force': np.arange(18, dtype=float) + self.calls}
        if self.failure == 'state' and self.calls:
            del obs['observation.state']
        if self.failure == 'state_shape' and self.calls:
            obs['observation.state'] = state[:8]
        return obs

    def step(self, action):
        self.calls += 1
        self.actions.append(action.copy())
        self.physical += np.sum(action) + random.random() + np.random.random()
        self.states.append(self.physical)
        info = {'requested_action': action.copy(),
                'applied_action': np.full(16, self.physical)}
        if self.failure in ('requested_action', 'applied_action'):
            del info[self.failure]
        if self.failure == 'applied_shape':
            info['applied_action'] = np.zeros(14)
        if self.failure == 'nan':
            info['applied_action'][0] = np.nan
        result = (self.obs(), 0., self.calls >= self.stop, False, info)
        self.results.append(result)
        return result


def reset(env, seed):
    env.physical = float(seed)
    return env.obs()


def seed_all():
    random.seed(4321)
    np.random.seed(4321)


class Acceptance(unittest.TestCase):
    def run_case(self, rows=32, steps=10, stop=100):
        seed_all()
        base_env, base_policy = Env(stop), Policy(rows)
        baseline = rollout.run_episode(base_env, base_policy, 12, reset, max_steps=steps)
        base_rng = (random.getstate(), np.random.get_state())
        seed_all()
        env, policy = Env(stop), Policy(rows)
        result, capture = run_captured(rollout.run_episode, env=env, policy=policy,
                                      reset=reset, seed=12, artifacts=ARTIFACTS, max_steps=steps)
        self.assertEqual(base_env.calls, env.calls)
        self.assertEqual(base_policy.calls, policy.calls)
        np.testing.assert_array_equal(base_env.actions, env.actions)
        np.testing.assert_array_equal(base_env.states, env.states)
        self.assertEqual(base_env.physical, env.physical)
        self.assertEqual(base_rng[0], random.getstate())
        current = np.random.get_state()
        self.assertEqual(base_rng[1][0], current[0])
        np.testing.assert_array_equal(base_rng[1][1], current[1])
        self.assertEqual(base_rng[1][2:], current[2:])
        self.assertEqual({k: v for k, v in baseline[0].items() if k != 'wall_s'},
                         {k: v for k, v in result[0].items() if k != 'wall_s'})
        np.testing.assert_array_equal(baseline[1], result[1])
        self.assertIs(policy.initial, policy.observed[0][0])
        for i, mapping in enumerate(capture.mapping):
            self.assertEqual((mapping['step'], mapping['obs_frame_before'], mapping['obs_frame_after']),
                             (i, i, i + 1))
            self.assertAlmostEqual(mapping['opening_before'], .2 + i * .01)
            self.assertAlmostEqual(mapping['opening_after'], .2 + (i + 1) * .01)
            self.assertEqual(mapping['run_id'], capture.run_id)
            self.assertIs(policy.observed[i + 1][0], env.results[i][0])
        self.assertNotIn('bridge', vars(policy))
        self.assertNotIn('adapter', vars(policy))
        self.assertIs(policy.bridge, SHARED_BRIDGE)
        self.assertNotIn('denormalize_action', vars(SHARED_BRIDGE))
        return capture

    def test_32_to_10(self):
        c = self.run_case()
        self.assertEqual([(x['predicted_rows'], x['executed_rows']) for x in c.chunks], [(32, 10)])

    def test_short_chunks(self):
        c = self.run_case(rows=3, steps=10)
        self.assertEqual([x['executed_rows'] for x in c.chunks], [3, 3, 3, 1])
        self.assertEqual((c.mapping[3]['chunk_id'], c.mapping[3]['chunk_row']), (1, 0))

    def test_early_termination(self):
        c = self.run_case(steps=30, stop=4)
        self.assertEqual(c.chunks[0]['executed_rows'], 4)

    def test_multichunk(self):
        c = self.run_case(steps=23)
        self.assertEqual([x['executed_rows'] for x in c.chunks], [10, 10, 3])

    def test_pollution_and_identity(self):
        seed_all()
        env, policy, c = Env(), Policy(), Capture(ARTIFACTS)
        obs = reset(env, 12)
        c.reset(obs)
        wrapped = CapturedPolicy(policy, c)
        returned = wrapped.chunk(obs, 0, 12)
        self.assertIs(returned, policy.results[0])
        np.testing.assert_array_equal(c.arrays['chunk_0_raw'], policy.raw_before)
        np.testing.assert_array_equal(c.arrays['chunk_0_denorm'], policy.raw_before + .125)
        expected = (policy.raw_before + .125)[0]
        expected[:, -1] = np.where(expected[:, -1] >= 0, 1., -1.)
        np.testing.assert_array_equal(c.arrays['chunk_0_rewritten'], expected)
        wrapped_env = CapturedEnv(env, c)
        result = wrapped_env.step(returned[0][0])
        self.assertIs(result, env.results[-1])
        self.assertIs(result[0], env.results[-1][0])
        self.assertIs(result[4], env.results[-1][4])
        # Evidence must remain independent of subsequent caller mutations.
        result[4]['requested_action'][:] = -999
        self.assertFalse(np.any(c.arrays['step_0_requested14'] == -999))

    def test_failures_and_restore(self):
        for failure in ('requested_action', 'applied_action', 'state', 'state_shape', 'applied_shape', 'nan'):
            with self.subTest(failure=failure):
                env, policy = Env(failure=failure), Policy()
                with self.assertRaises(ValueError) as caught:
                    run_captured(rollout.run_episode, env=env, policy=policy, reset=reset,
                                 seed=1, artifacts=ARTIFACTS, max_steps=1)
                self.assertEqual(caught.exception.execution_capture.status, 'failed')
                self.assertNotIn('bridge', vars(policy))
                self.assertNotIn('adapter', vars(policy))
        for failure, error in (('throw', RuntimeError), ('bad_adapter', ValueError)):
            policy = Policy(failure=failure)
            policy.bridge, policy.adapter = Bridge(), Adapter()
            old = (policy.bridge, policy.adapter)
            with self.assertRaises(error):
                run_captured(rollout.run_episode, env=Env(), policy=policy, reset=reset,
                             seed=1, artifacts=ARTIFACTS, max_steps=1)
            self.assertIs(policy.bridge, old[0])
            self.assertIs(policy.adapter, old[1])

    def test_wrong_mapping(self):
        c, p, e = Capture(ARTIFACTS), Policy(), Env()
        obs = reset(e, 1)
        c.reset(obs)
        wrapped = CapturedPolicy(p, c)
        with self.assertRaises(ValueError):
            wrapped.chunk(obs, 1, 1)
        actions = wrapped.chunk(obs, 0, 1)[0]
        with self.assertRaises(ValueError):
            CapturedEnv(e, c).step(actions[0] + 1)
        self.assertEqual(e.calls, 0)
        c.frame = 1
        with self.assertRaises(ValueError):
            CapturedEnv(e, c).step(actions[0])

    def test_stage_dimensions_finiteness(self):
        for name in ('raw', 'denorm', 'rewritten', 'adapter'):
            for fault in ('shape', 'nan'):
                with self.subTest(stage=name, fault=fault):
                    c = Capture(ARTIFACTS)
                    c.reset({'observation.state': np.zeros(16)})
                    stages = dict(raw=np.zeros((1, 3, 7)), denorm=np.zeros((1, 3, 7)),
                                  rewritten=np.zeros((3, 7)), adapter=np.zeros((3, 14)))
                    if fault == 'shape':
                        stages[name] = stages[name][..., :-1]
                    else:
                        stages[name].flat[0] = np.nan
                    with self.assertRaises(ValueError):
                        c.chunk(0, **stages)

    def test_metadata_save_read_exclusive(self):
        c = self.run_case()
        path = ROOT / ('capture_cpu_acceptance_' + uuid.uuid4().hex + '.npz')
        c.save(path)
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data['metadata']))
            self.assertEqual(meta['run_id'], c.run_id)
            self.assertEqual(meta['status'], 'complete')
            for entries in meta['artifacts'].values():
                for entry in entries:
                    self.assertEqual(hashlib.sha256(Path(entry['path']).read_bytes()).hexdigest(), entry['sha256'])
            self.assertEqual(meta['steps'], c.mapping)
            np.testing.assert_array_equal(data['chunk_0_raw'], c.arrays['chunk_0_raw'])
        before = path.read_bytes()
        with self.assertRaises(FileExistsError):
            c.save(path)
        self.assertEqual(before, path.read_bytes())
        print('Saved acceptance artifact (synthetic weights/config):', path)

    def test_required_hashes(self):
        with self.assertRaises(ValueError):
            Capture({'source': [__file__]})
        with self.assertRaises(FileNotFoundError):
            Capture(dict(ARTIFACTS, weights=[str(ROOT / 'nonexistent_weights')]))

    def test_actual_policy_chunk_source_cpu(self):
        # Compile ONLY the real chunk method, without importing policy40k/torch.
        # no_grad is a no-op for this NumPy-only model stub, not real Torch coverage.
        source_path = ORIGINAL / 'policy40k.py'
        tree = ast.parse(source_path.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ForcePolicy')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'chunk')
        namespace = {'np': np, 'time': time, 'torch': SimpleNamespace(no_grad=nullcontext),
                     'noise_seed': lambda seed, calls: seed + calls}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), namespace)
        actual_chunk = namespace['chunk']

        class ActualPolicy(Policy):
            chunk = actual_chunk

            def __init__(self):
                super().__init__()
                self.stream = force.ForceStream()
                self.action_horizon = 32
                self.processor = None
                self.gripper_mode = 'pm1'
                self.model = SimpleNamespace(infer_action=lambda **kw: {
                    'action': np.random.uniform(-.8, .8, (1, 32, 7)),
                    'force_future_newton': np.zeros((1, 32, 6))})

            def _infer_kwargs(self, obs, history, mask, ns):
                return {}

            def observe(self, obs, frame):
                self.stream.push(obs['observation.force'])

        def execute(wired):
            seed_all()
            env, policy = Env(), ActualPolicy()
            if wired:
                result, capture = run_captured(rollout.run_episode, env=env, policy=policy,
                    reset=reset, seed=12, artifacts=dict(ARTIFACTS,
                        source=ARTIFACTS['source'] + [str(source_path)]), max_steps=23)
                self.assertEqual([x['executed_rows'] for x in capture.chunks], [10, 10, 3])
            else:
                result = rollout.run_episode(env, policy, 12, reset, max_steps=23)
            return env, policy, result, random.getstate(), np.random.get_state()

        base, wired = execute(False), execute(True)
        np.testing.assert_array_equal(base[0].actions, wired[0].actions)
        np.testing.assert_array_equal(base[0].states, wired[0].states)
        np.testing.assert_array_equal(base[2][1], wired[2][1])
        self.assertEqual(base[1].calls, wired[1].calls)
        self.assertEqual(base[3], wired[3])
        np.testing.assert_array_equal(base[4][1], wired[4][1])
        self.assertEqual(base[4][2:], wired[4][2:])
        self.assertNotIn('bridge', vars(wired[1]))
        self.assertNotIn('adapter', vars(wired[1]))

    def test_torch_rng_if_available(self):
        if importlib.util.find_spec('torch') is None:
            self.skipTest('Torch not installed: Torch tensor / RNG not tested; no dependencies installed')
        import torch
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        from execution_chain_capture import copy_cpu
        np.testing.assert_array_equal(copy_cpu(torch.ones(2)), np.ones(2))
        self.run_case()
        self.assertTrue(torch.equal(before, torch.get_rng_state()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
