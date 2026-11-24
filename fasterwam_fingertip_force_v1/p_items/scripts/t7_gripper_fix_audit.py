"""Gripper fix audit: raw expert opening -> pm1 label -> env sign -> training balance.

The V2 labels are pm1: +1 = close, -1 = open. This script checks:
  1) label statistics over all 845 episodes (close fraction, transitions, pulses)
  2) raw action[7] opening vs the pm1 label
  3) the env grip sign in joint_position and cartesian_delta modes
  4) the training code's balanced sampler / close_frac metric semantics
"""
import json
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
LABELS = P + '/A3_845_7D_LABELS_V2.npz'
OUT = P + '/GRIPPER_FIX_AUDIT.json'
DATA_PY = '/workspace/_t7_data.py'
TRAIN_PY = '/workspace/_t7_train.py'


def runs(mask):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        out.append(j - i)
        i = j
    return out


def env_signs():
    from a3_dual_arm_sim import A3CookieTransferEnv
    out = {}
    env = A3CookieTransferEnv(action_mode='joint_position', render_mode=None, render_cameras=False)
    u = getattr(env, 'unwrapped', env)
    env.reset(seed=123)
    home = np.asarray(u.DEPLOYMENT_HOME, np.float64).copy()
    for grip, name in ((1.0, 'joint_grip_1'), (0.0, 'joint_grip_0')):
        a = home.copy()
        a[7] = grip
        for _ in range(8):
            obs, r, term, trunc, info = env.step(a)
        out[name] = float(u._gripper_state('L')[0])
    env.close()
    env = A3CookieTransferEnv(action_mode='cartesian_delta', render_mode=None, render_cameras=False)
    u = getattr(env, 'unwrapped', env)
    env.reset(seed=123)
    for grip, name in ((1.0, 'cart_grip_plus1'), (-1.0, 'cart_grip_minus1')):
        a = np.zeros(14, np.float64)
        a[6] = grip
        for _ in range(8):
            obs, r, term, trunc, info = env.step(a)
        out[name] = float(u._gripper_state('L')[0])
    env.close()
    return out


def main():
    t = pq.read_table(PARQ, columns=['episode_index', 'action'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E, A = E[order], A[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]
    z = np.load(LABELS)
    L = z['label'].astype(np.float64)
    lab_start = z['ep_start_label'].astype(np.int64)
    lab_len = z['ep_len'].astype(np.int64)

    rows = []
    for k, ep in enumerate(ids):
        s, e = int(st[k]), int(en[k])
        n = e - s
        ls = int(lab_start[ep])
        ll = int(lab_len[ep])
        raw = A[s:e, 7]
        lab = L[ls:ls + ll, 6]
        close = lab > 0
        trans = np.flatnonzero(np.diff(lab) != 0) + 1
        pulses = 0
        for i in range(len(lab)):
            left = (i == 0) or (lab[i] != lab[i - 1])
            right = (i == len(lab) - 1) or (lab[i] != lab[i + 1])
            if left and right:
                pulses += 1
        rows.append({
            'episode': ep, 'frames': n, 'label_rows': ll,
            'task_class': '2x10' if ep >= 749 else '2x5',
            'label_close_frac': float(close.mean()) if len(close) else None,
            'raw_close_cmd_frac': float((raw <= 0.6).mean()),
            'raw_open_cmd_frac': float((raw >= 0.8).mean()),
            'raw_deadband_frac': float(((raw > 0.6) & (raw < 0.8)).mean()),
            'transitions': int(len(trans)),
            'pulses_1step': int(pulses),
            'open_runs': runs(~close), 'close_runs': runs(close),
            'first_close': int(np.argmax(close)) if close.any() else None,
            'first_open_after_close': None,
            'always_closed': bool(close.all()), 'always_open': bool((~close).all()),
            'naive_threshold_mismatch': float((((raw[:ll] <= 0.7) != close)).mean()),
        })
        if close.any():
            first_close = int(np.argmax(close))
            after = np.flatnonzero(~close[first_close:])
            if len(after):
                rows[-1]['first_open_after_close'] = int(first_close + after[0])

    def agg(sel):
        x = [r for r in rows if sel(r)]
        if not x:
            return None
        cf = [r['label_close_frac'] for r in x]
        op = [r for r in x if r['open_runs']]
        return {
            'episodes': len(x),
            'close_frac_mean': float(np.mean(cf)),
            'close_frac_median': float(np.median(cf)),
            'close_frac_min': float(np.min(cf)),
            'close_frac_max': float(np.max(cf)),
            'episodes_close_gt_0.95': int(sum(1 for v in cf if v > 0.95)),
            'episodes_close_gt_0.97': int(sum(1 for v in cf if v > 0.97)),
            'episodes_close_eq_1.0': int(sum(1 for v in cf if v >= 0.9999)),
            'always_closed_episodes': int(sum(1 for r in x if r['always_closed'])),
            'always_open_episodes': int(sum(1 for r in x if r['always_open'])),
            'transitions_mean': float(np.mean([r['transitions'] for r in x])),
            'transitions_median': float(np.median([r['transitions'] for r in x])),
            'episodes_with_open': int(len(op)),
            'open_run_len_median': float(np.median([np.median(r['open_runs']) for r in op])) if op else None,
            'close_run_len_median': float(np.median([np.median(r['close_runs']) for r in x if r['close_runs']])) if x else None,
            'pulses_1step_total': int(sum(r['pulses_1step'] for r in x)),
            'raw_close_cmd_frac_mean': float(np.mean([r['raw_close_cmd_frac'] for r in x])),
            'naive_mismatch_mean': float(np.mean([r['naive_threshold_mismatch'] for r in x])),
        }

    audit = {'worker': 'supervisor', 'ts': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
             'label_semantics': 'pm1: +1 close (raw opening <=0.6), -1 open (raw opening >=0.8)',
             'by_class': {'2x5': agg(lambda r: r['task_class'] == '2x5'),
                          '2x10': agg(lambda r: r['task_class'] == '2x10'),
                          'all': agg(lambda r: True)},
             'first10': rows[:10],
             'env_grip_signs': env_signs(),
             'training_code_check': {}}
    try:
        txt = open(DATA_PY).read().splitlines()
        hits = [(i + 1, l.strip()) for i, l in enumerate(txt) if ('phase_open' in l or 'phase_rise' in l or 'open_episodes' in l)]
        audit['training_code_check']['t7_data_phase_lines'] = hits[:12]
    except Exception as exc:
        audit['training_code_check']['t7_data_error'] = str(exc)
    try:
        txt = open(TRAIN_PY).read().splitlines()
        hits = [(i + 1, l.strip()) for i, l in enumerate(txt) if 'close_frac' in l]
        audit['training_code_check']['t7_train_close_frac_lines'] = hits[:6]
    except Exception as exc:
        audit['training_code_check']['t7_train_error'] = str(exc)
    audit['training_code_check']['finding'] = (
        'labels are pm1 (+1 close). t7_data.py phase_open uses g>=0.8, which selects +1 close, '
        'so open_episodes / phase_rise are close frames/transitions, not open. '
        't7_train.py close_frac uses action[:,:,6] <= 0.6, which selects -1 open, so the logged '
        'close_frac is an open fraction. Both are inverted relative to their names.')
    json.dump(audit, open(OUT, 'w'), indent=1)
    print(json.dumps(audit['by_class'], indent=1)[:4000], flush=True)
    print(json.dumps(audit['env_grip_signs'], indent=1), flush=True)
    print(json.dumps(audit['training_code_check'], indent=1)[:1500], flush=True)


if __name__ == '__main__':
    main()