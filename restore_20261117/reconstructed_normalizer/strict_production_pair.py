"""Opt-in, ten-step pair; CPU preflight is mandatory and never loads weights.

RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL; original_stats_sha256=UNKNOWN.
Synthetic regression fixtures are NOT production asset acceptance.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import diag_reconstructed_40k as D

HERE = Path(__file__).resolve().parent


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_paths():
    # Local archived code is usable for hashing; asset paths are NEVER substituted.
    original = D.CAPTURE / 'execution_chain_capture.py'
    if not original.is_file():
        original = HERE.parent / 'deploy' / 'capture' / 'execution_chain_capture.py'
    return {'original': original, 'fixed': HERE / 'execution_chain_capture_fixed.py',
            'launcher': Path(__file__).resolve()}


def preflight():
    paths = source_paths()
    report = {**D.MARKERS, 'pass': False, 'status': 'FAIL',
              'scope': 'REAL_ASSETS_CPU_PREFLIGHT', 'artifacts': copy.deepcopy(D.artifacts()),
              'source_identity': {}, 'errors': []}
    for name, path in paths.items():
        entry = {'path': str(path)}
        try:
            entry.update(sha256=D.sha256_file(path), bytes=path.stat().st_size)
        except OSError as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            report['errors'].append(entry['error'])
        report['source_identity'][name] = entry
    try:
        original = load_file('_strict_original_cpu', paths['original'])
        # The unmodified archived function streams every actual asset byte.
        report['artifact_hashes'] = original.hash_files(report['artifacts'])
    except Exception as exc:
        report['errors'].append(f'{type(exc).__name__}: {exc}')
    report['pass'] = not report['errors']
    report['status'] = 'PASS' if report['pass'] else 'FAIL'
    return report


def load_runtime():
    # This function is unreachable until successful CPU preflight AND explicit opt-in.
    import torch
    import run_forensics40k as F
    E, factory, W, fw = D.setup_protocol()
    EC = load_file('_strict_evaluation', D.CAPTURE / 'evaluation_capture.py')
    FIX = load_file('_strict_fixed', source_paths()['fixed'])
    EC.Capture = FIX.Capture
    return SimpleNamespace(torch=torch, F=F, E=E, factory=factory, W=W, fw=fw, EC=EC)


def make_witnessed_runner(original_runner, EC, torch, out, state, artifacts, hashes):
    def witnessed_runner(env, policy, seed, reset, **kwargs):
        frames, infos = [], []
        original_step = env.step
        base = getattr(env, 'unwrapped', env)

        def snapshot(obs):
            row = {k: np.array(v, copy=True) for k, v in obs.items()
                   if isinstance(v, (np.ndarray, float, int, bool, np.number))}
            for name in ('qpos', 'qvel', 'ctrl', 'act', 'qacc', 'sensordata'):
                row['mujoco.' + name] = np.array(getattr(base.data, name), copy=True)
            frames.append(row)

        def step(action, **kw):
            result = original_step(action, **kw)
            snapshot(result[0])
            infos.append(result[4])
            return result

        def reset_seen(e, s):
            obs = reset(e, s)
            snapshot(obs)
            return obs

        # Capture construction can itself fail; step restoration still runs.
        gpu_pre = torch.cuda.get_rng_state_all()
        env.step = step
        try:
            result, capture = EC.run_captured(
                original_runner, env=env, policy=policy, reset=reset_seen,
                seed=seed, artifacts=artifacts, **kwargs)
            if capture.artifacts != hashes:
                raise RuntimeError('capture artifact identity differs from CPU preflight')
        finally:
            env.step = original_step
            arrays = {k: np.stack([row[k] for row in frames]) for k in frames[0]} if frames else {}
            arrays.update({'cuda_rng_pre_' + str(i): x.cpu().numpy() for i, x in enumerate(gpu_pre)})
            arrays.update({'cuda_rng_post_' + str(i): x.cpu().numpy()
                           for i, x in enumerate(torch.cuda.get_rng_state_all())})
            tag = state['tag']
            np.savez_compressed(out / ('witness_' + tag + '.npz'), **arrays)
            (out / ('info_' + tag + '.json')).write_text(json.dumps(
                infos, sort_keys=True, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v)))
        capture.save(str(out / ('execution_' + tag + '.npz')))
        return result
    return witnessed_runner


def compare_outputs(out, A, B):
    comparison = D.compare_pair(A, B)
    checks = {}
    for prefix in ('witness', 'execution'):
        with np.load(out / (prefix + '_A.npz'), allow_pickle=False) as a, np.load(
                out / (prefix + '_B.npz'), allow_pickle=False) as b:
            checks[prefix + '_keys'] = set(a.files) == set(b.files)
            for key in set(a.files) | set(b.files):
                if key != 'metadata':
                    checks[prefix + ':' + key] = key in a and key in b and np.array_equal(a[key], b[key])
    checks['info_exact'] = (out / 'info_A.json').read_bytes() == (out / 'info_B.json').read_bytes()
    checks['recorded_comparison'] = comparison['no_side_effect_bit_exact']
    checks['cpu_rng'] = all(all(comparison[key].values()) for key in ('rng_pre_equal', 'rng_post_equal'))
    return {'pass': all(checks.values()), 'checks': checks, 'comparison': comparison}


def run_pair(pre, out):
    if not pre['pass']:
        raise RuntimeError('CPU preflight failed; GPU construction forbidden')
    r = load_runtime()
    args = SimpleNamespace(cpu=False, gripper_mode='pm1', max_steps=10, replan=10,
                           allow_invalid_scene=False)
    policy, doc, calib, facts = D.build_policy_reconstructed(r.E, r.W, args)
    PV = r.F.install_cv2_recorder()
    original_runner = r.E.run_episode
    state = {'tag': 'A'}
    # One shared artifact mapping and one preflight identity for both arms.
    r.E.run_episode = make_witnessed_runner(original_runner, r.EC, r.torch, out, state,
                                            pre['artifacts'], pre['artifact_hashes'])

    def reseed():
        r.torch.manual_seed(12)
        random.seed(12)
        np.random.seed(12)

    try:
        reseed()
        A = D.run_one(r.E, r.factory, policy, args, 12, record_chain=False)
        D.save_run(out, 12, 'strict_A', A)
        state['tag'] = 'B'
        reseed()
        B = r.F.run_one_forensics(r.E, r.factory, policy, r.W, calib, args, 12, out, PV)
        D.save_run(out, 12, 'strict_B', B)
        report = compare_outputs(out, A, B)
        report.update(protocol=facts, fw_import=r.fw,
                      driver_sha256=D.sha256_file(Path(r.F.__file__)),
                      comparison_basis='Common fixed capture and state witness; only B enables forensics')
        return report
    finally:
        r.E.run_episode = original_runner


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight-only', action='store_true')
    mode.add_argument('--run-pair', action='store_true', help='explicit opt-in: seed 12, exactly 10 steps per arm')
    parser.add_argument('--out', type=Path, default=HERE / 'strict_production_out')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    pre = preflight()
    write_report(args.out / 'preflight.json', pre)
    if not pre['pass']:
        print(json.dumps({'status': 'FAIL', 'phase': 'CPU_PREFLIGHT', 'errors': pre['errors']}))
        return 1
    if not args.run_pair:
        print(json.dumps({'status': 'PASS', 'phase': 'CPU_PREFLIGHT', 'pair_run': False}))
        return 0
    report = {**D.MARKERS, 'preflight': pre, 'seed': 12, 'max_steps': 10, 'pass': False}
    try:
        report.update(run_pair(pre, args.out))
    except BaseException as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        write_report(args.out / 'strict_pair.json', report)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        print(json.dumps({'status': 'FAIL', 'error': report['error']}))
        return 1
    report['status'] = 'PASS' if report['pass'] else 'FAIL'
    write_report(args.out / 'strict_pair.json', report)
    return 0 if report['pass'] else 2


if __name__ == '__main__':
    sys.exit(main())
