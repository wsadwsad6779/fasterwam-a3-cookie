import os, sys, json, time, traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np
import pyarrow.parquet as pq

OUT = '/workspace/fwv1/p_items/P5_REPLAY_EXTENDED.json'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
EPISODES = [10, 100, 300, 500, 700]
MAX_STEPS = 600
CLOSED_TH = 0.7
OPEN_TH = 0.85


def v2(g):
    n = len(g)
    lab = np.zeros(n)
    for i in range(n):
        if g[i] <= 0.6:
            lab[i] = 1
        elif g[i] >= 0.8:
            lab[i] = -1
        else:
            lab[i] = lab[i - 1] if i else 1
    for _ in range(3):
        i = 0
        while i < n:
            j = i
            while j < n and lab[j] == lab[i]:
                j += 1
            if j - i < 5:
                lab[i:j] = lab[i - 1] if i > 0 else (lab[j] if j < n else lab[i])
            i = j
    return lab


def load_actions():
    t = pq.read_table(PARQ, columns=['episode_index', 'action'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E = E[order]
    A = A[order]
    b = np.flatnonzero(np.diff(E)) + 1
    return np.r_[0, b], np.r_[b, len(E)], A


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    st, en, A = load_actions()
    eps = []
    for ep in EPISODES:
        s, e = int(st[ep]), int(en[ep])
        act = A[s:e]
        env = A3CookieTransferEnv(action_mode='joint_position', render_cameras=False)
        u = getattr(env, 'unwrapped', env)
        obs, _ = env.reset(seed=20000 + ep, options={'randomize_cookies': False, 'randomize_boxes': False})
        home = np.asarray(u.DEPLOYMENT_HOME, dtype=np.float64)
        right_arm = home[8:15].copy()
        right_grip = float(home[15])
        rec = {'episode': ep, 'dataset_frames': int(e - s)}
        try:
            ops = []
            cmds = []
            info = {}
            for t in range(min(len(act), MAX_STEPS)):
                a16 = np.r_[act[t, :7], act[t, 7], right_arm, right_grip]
                obs, r, term, trunc, info = env.step(a16)
                ops.append(float(u._gripper_state('L')[0]))
                cmds.append(float(act[t, 7]))
                if term or trunc:
                    break
            ops = np.asarray(ops)
            cmds = np.asarray(cmds)
            lab = v2(cmds)
            close = lab > 0
            openm = lab < 0
            best = None
            for lag in range(0, 16):
                if lag == 0:
                    cl, om, o = close, openm, ops
                else:
                    cl = close[:-lag]
                    om = openm[:-lag]
                    o = ops[lag:]
                if not cl.any() or not om.any():
                    continue
                acc = 0.5 * (float((o[cl] <= CLOSED_TH).mean()) + float((o[om] >= OPEN_TH).mean()))
                if best is None or acc > best['acc']:
                    best = {'lag': lag, 'acc': acc,
                            'close_ok': float((o[cl] <= CLOSED_TH).mean()),
                            'open_ok': float((o[om] >= OPEN_TH).mean())}
            trans = np.flatnonzero(np.diff(lab) != 0) + 1
            matched = 0
            for i in trans:
                want_close = lab[i] > 0
                pre = float(np.median(ops[max(0, i - 5):i])) if i > 0 else float(ops[0])
                win = ops[i:min(i + 16, len(ops))]
                if len(win):
                    if want_close and float(win.min()) <= pre - 0.1:
                        matched += 1
                    if (not want_close) and float(win.max()) >= pre + 0.1:
                        matched += 1
            rec.update({'steps': len(ops), 'best': best, 'transitions': int(len(trans)),
                        'transitions_confirmed': int(matched),
                        'label_pulses': int(sum(1 for i in range(len(lab))
                                                if (i == 0 or lab[i] != lab[i - 1]) and
                                                (i == len(lab) - 1 or lab[i + 1] != lab[i]))),
                        'final_info': {k: info.get(k) for k in ('cookies_in_target', 'cookies_in_source')},
                        'ok': bool(best is not None and best['acc'] >= 0.8 and
                                   len(trans) > 0 and matched / len(trans) >= 0.8)})
        except Exception:
            rec['error'] = traceback.format_exc()[:1000]
            rec['ok'] = False
        try:
            env.close()
        except Exception:
            pass
        eps.append(rec)
        print(json.dumps({k: rec.get(k) for k in ['episode', 'steps', 'best', 'transitions',
                                                   'transitions_confirmed', 'label_pulses', 'ok']}), flush=True)
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'episodes': eps,
           'n_ok': sum(1 for x in eps if x.get('ok')),
           'verdict': 'pass' if all(x.get('ok') for x in eps) else 'fail',
           'note': 'metrics-only independent replays (no camera rendering); same action mapping and thresholds as P5_REPLAY_VIDEO.json'}
    json.dump(res, open(OUT, 'w'), indent=1)
    print('DONE ' + res['verdict'])


if __name__ == '__main__':
    main()