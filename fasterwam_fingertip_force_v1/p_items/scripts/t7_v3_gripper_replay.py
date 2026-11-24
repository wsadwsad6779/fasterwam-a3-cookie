"""Verify V3 continuous opening labels drive the real MuJoCo gripper (open/close/partial)
before any retraining."""
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np

P = '/workspace/fwv1/p_items'
NPZ = P + '/A3_845_7D_LABELS_V3.npz'
OUT = P + '/GRIPPER_V3_ACCEPTANCE.json'
MD = P + '/GRIPPER_V3_ACCEPTANCE.md'
EPISODES = [100, 422, 700]


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    z = np.load(NPZ)
    L = z['label'].astype(np.float64)
    ls = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'labels': NPZ, 'episodes': []}
    for ep in EPISODES:
        k = ep
        s = int(ls[k]); n = min(int(el[k]), 600)
        cmd = L[s:s + n, 6]
        env = A3CookieTransferEnv(action_mode='cartesian_delta', render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        env.reset(seed=10000 + ep)
        ops = []
        try:
            for i in range(n):
                gcart = float(np.clip(2.0 * cmd[i] - 1.0, -1.0, 1.0))
                a = np.r_[0, 0, 0, 0, 0, 0, gcart, 0, 0, 0, 0, 0, 0, 1.0]
                obs, r, term, trunc, info = env.step(a)
                ops.append(float(u._gripper_state('L')[0]))
                if term or trunc:
                    break
        finally:
            try:
                env.close()
            except Exception:
                pass
        ops = np.asarray(ops)
        m = min(len(cmd), len(ops))
        cmd, ops = cmd[:m], ops[:m]
        best = None
        for lag in range(0, 16):
            c = cmd[:m - lag] if lag else cmd
            o = ops[lag:] if lag else ops
            if len(c) < 10:
                continue
            rmse = float(np.sqrt(np.mean((c - o) ** 2)))
            if best is None or rmse < best['rmse']:
                best = {'lag': lag, 'rmse': rmse,
                        'corr': float(np.corrcoef(c, o)[0, 1]) if np.std(c) > 0 and np.std(o) > 0 else None,
                        'open_ge08_actual_mean': float(o[c >= 0.8].mean()) if (c >= 0.8).any() else None,
                        'partial_actual_mean': float(o[(c > 0.35) & (c < 0.8)].mean()) if ((c > 0.35) & (c < 0.8)).any() else None,
                        'closed_le035_actual_mean': float(o[c <= 0.35].mean()) if (c <= 0.35).any() else None}
        closed = cmd <= 0.6
        openm = cmd >= 0.8
        rises = np.flatnonzero(np.diff(openm.astype(int)) == 1) + 1
        falls = np.flatnonzero(np.diff(openm.astype(int)) == -1) + 1
        close_segments = int((np.diff(closed.astype(int)) == 1).sum())
        cycles = 0
        for i in range(1, len(cmd) - 1):
            if closed[i] and not closed[i - 1]:
                # find a rise to open after this close, then another close
                after = np.flatnonzero(openm[i:])
                if len(after):
                    j = i + int(after[0])
                    if closed[j:].any():
                        cycles += 1
        res['episodes'].append({'episode': ep, 'steps': m,
                                'cmd_open_frac': float(openm.mean()),
                                'cmd_partial_frac': float(((cmd > 0.35) & (cmd < 0.8)).mean()),
                                'cmd_closed_frac': float((cmd <= 0.35).mean()),
                                'rise_events': int(len(rises)), 'fall_events': int(len(falls)),
                                'close_segments': close_segments,
                                'complete_close_open_reclose_cycles': int(cycles),
                                'tracking': best,
                                'v3_ok': bool(best is not None and best['rmse'] < 0.12
                                              and (best['open_ge08_actual_mean'] or 0) > 0.7
                                              and (best['closed_le035_actual_mean'] is not None))})
    res['verdict'] = 'pass' if all(e['v3_ok'] for e in res['episodes']) else 'fail'
    json.dump(res, open(OUT, 'w'), indent=1)
    lines = ['# GRIPPER V3 ACCEPTANCE (continuous commanded opening)', '']
    for e in res['episodes']:
        lines.append('- ep%d: open %.2f / partial %.2f / closed %.2f, rises %d, falls %d, cycles %d, track RMSE %.3f corr %s lag %s' % (
            e['episode'], e['cmd_open_frac'], e['cmd_partial_frac'], e['cmd_closed_frac'],
            e['rise_events'], e['fall_events'], e['complete_close_open_reclose_cycles'],
            e['tracking']['rmse'], e['tracking']['corr'], e['tracking']['lag']))
    lines += ['', 'Verdict: **%s**' % res['verdict']]
    open(MD, 'w').write('\n'.join(lines))
    print(json.dumps({'verdict': res['verdict'], 'episodes': res['episodes']})[:1200])


if __name__ == '__main__':
    main()