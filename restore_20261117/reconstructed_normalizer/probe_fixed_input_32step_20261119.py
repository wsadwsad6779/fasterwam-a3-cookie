"""Fixed-input 32-step plan comparison: base 40k (A) vs step_100 action-expert delta (C).

Both arms see the identical reset observation of seed 29 and the same diffusion noise seed
(chunk call #1). We capture the raw 32x7 model actions (pre-Adapter), the Adapter env actions
and the P3 force forecast, then rebuild the second arm in the same process.
"""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'c_class_eval_20261119' / 'fixed_input_32step'
CKPT = Path(os.environ.get('C_CKPT', str(
    ROOT / 'model_repair_runs' / 'lora_300' / 'step_000100.pt')))
SEED = int(os.environ.get('C_SEED', '29'))

E, factory, W, fw = D.setup_protocol()
import torch


def build():
    policy, doc, calib, facts = D.build_policy_reconstructed(
        E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
    return policy, doc, calib, facts


def load_delta(policy, path):
    sd = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    ck = sd['delta'] if 'delta' in sd else sd['model']
    own = policy.model.state_dict()
    bad = [k for k in ck if k not in own or tuple(own[k].shape) != tuple(ck[k].shape)]
    if bad:
        raise SystemExit('checkpoint mismatch: %s' % bad[:3])
    policy.model.load_state_dict({k: v for k, v in ck.items()},
                                 strict=('delta' not in sd))
    policy.model.eval()
    return {'kind': 'DELTA_ONLY' if 'delta' in sd else 'FULL', 'step': int(sd.get('step', -1)),
            'keys': len(ck)}


def run(policy, tag, do_load=None):
    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=False, gripper_mode='pm1', max_steps=10, replan=10, allow_invalid_scene=False))
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    obs = reset(env, SEED)
    load_info = do_load(policy) if do_load else None
    policy.start_episode(obs, SEED)
    policy.observe(obs, 0)
    raw = []
    real = policy.adapter.convert

    def spy(a, rot):
        raw.append(np.asarray(a, dtype=np.float64).copy())
        return real(a, rot)

    policy.adapter.convert = spy
    actions, pf, meta = policy.chunk(obs, frame=0, seed=SEED)
    policy.adapter.convert = real
    if hasattr(env, 'close'):
        env.close()
    return {'raw': np.stack(raw), 'env': np.asarray(actions, dtype=np.float64),
            'pred_force': np.asarray(pf, dtype=np.float64), 'meta': meta,
            'load': load_info}


def summarise(a, c):
    ra, rc = a['raw'], c['raw']
    out = {'A_meta': a['meta'], 'C_meta': c['meta'],
           'same_noise_seed': a['meta']['noise_seed'] == c['meta']['noise_seed']}
    dt = (np.diff(ra, axis=0) if len(ra) > 1 else np.zeros((0, 7)))
    dc = (np.diff(rc, axis=0) if len(rc) > 1 else np.zeros((0, 7)))
    out['plan_smoothness'] = {'A_mean_row_delta': float(np.abs(dt).mean()),
                              'C_mean_row_delta': float(np.abs(dc).mean())}
    out['per_dim'] = {}
    for i, name in enumerate(['dx', 'dy', 'dz', 'rx', 'ry', 'rz', 'grip']):
        x, y = ra[:, i], rc[:, i]
        out['per_dim'][name] = {
            'A_mean': float(x.mean()), 'C_mean': float(y.mean()),
            'A_absmax': float(np.abs(x).max()), 'C_absmax': float(np.abs(y).max()),
            'corr': float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else None}
    out['grip_sign_change_row_A'] = int((np.sign(ra[1:, 6]) != np.sign(ra[:-1, 6])).sum())
    out['grip_sign_change_row_C'] = int((np.sign(rc[1:, 6]) != np.sign(rc[:-1, 6])).sum())
    out['grip_A'] = np.sign(ra[:, 6]).astype(int).tolist()
    out['grip_C'] = np.sign(rc[:, 6]).astype(int).tolist()
    out['pred_force_absmax'] = {'A': float(np.abs(a['pred_force']).max()),
                                'C': float(np.abs(c['pred_force']).max())}
    out['pred_force_mean_abs_diff'] = float(
        np.abs(a['pred_force'] - c['pred_force']).mean())
    out['env_translation_absmax'] = {'A': float(np.abs(a['env'][:, :3]).max()),
                                     'C': float(np.abs(c['env'][:, :3]).max())}
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print('[A] building base 40k', flush=True)
    pa, _, _, facts = build()
    a = run(pa, 'A')
    del pa
    torch.cuda.empty_cache()
    print('[C] building base and loading', CKPT, flush=True)
    pc, _, _, _ = build()
    c = run(pc, 'C', do_load=lambda p: load_delta(p, CKPT))
    del pc
    torch.cuda.empty_cache()
    out = {**D.MARKERS, 'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
           'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA', 'seed': SEED,
           'checkpoint': str(CKPT), 'C_load': c['load'], **summarise(a, c)}
    np.savez_compressed(OUT / 'raw_plans.npz', A_raw=a['raw'], C_raw=c['raw'],
                        A_pred_force=a['pred_force'], C_pred_force=c['pred_force'],
                        A_env=a['env'], C_env=c['env'])
    (OUT / 'summary.json').write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ('same_noise_seed', 'plan_smoothness', 'per_dim',
                                          'grip_sign_change_row_A', 'grip_sign_change_row_C',
                                          'pred_force_absmax')}, indent=2)[:2600])
    print('FIXED_INPUT_DONE', flush=True)


if __name__ == '__main__':
    main()
