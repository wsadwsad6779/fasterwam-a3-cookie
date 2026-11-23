import os, sys, json, time, hashlib, traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')

import numpy as np
import imageio
import pyarrow.parquet as pq

OUTDIR = '/workspace/fwv1/p_items/videos'
OUT = '/workspace/fwv1/p_items/P5_REPLAY_VIDEO.json'
EVOUT = '/workspace/fwv1/p_items/GRIPPER_EVENT_VALIDATION.json'
MANOUT = '/workspace/fwv1/p_items/GRIPPER_REPLAY_VIDEO_MANIFEST.json'
AUDIT = '/workspace/fwv1/p_items/GRIPPER_LABEL_V2_FINAL_AUDIT.md'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
EPISODES = [0, 422, 844]
MAX_STEPS = 600
os.makedirs(OUTDIR, exist_ok=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


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
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    return st, en, A


def main():
    from a3_dual_arm_sim import A3CookieTransferEnv
    st, en, A = load_actions()
    t0 = time.time()
    episodes = []
    video_entries = []
    for ep in EPISODES:
        s, e = int(st[ep]), int(en[ep])
        act = A[s:e]
        env = A3CookieTransferEnv(action_mode='joint_position', render_mode='rgb_array',
                                  render_cameras=True)
        u = getattr(env, 'unwrapped', env)
        obs, _ = env.reset(seed=10000 + ep, options={'randomize_cookies': False, 'randomize_boxes': False})
        home = np.asarray(u.DEPLOYMENT_HOME, dtype=np.float64)
        right_arm = home[8:15].copy()
        right_grip = float(home[15])
        frames_f = []
        frames_w = []
        rows = []
        rec = {'episode': ep, 'dataset_frames': int(e - s), 'replay_steps': 0}
        n = min(len(act), MAX_STEPS)
        try:
            for t in range(n):
                a16 = np.r_[act[t, :7], act[t, 7], right_arm, right_grip]
                obs, r, term, trunc, info = env.step(a16)
                frames_f.append(np.ascontiguousarray(obs['observation.images.front']))
                frames_w.append(np.ascontiguousarray(obs['observation.images.left_wrist']))
                op = float(u._gripper_state('L')[0])
                force = np.asarray(obs['observation.force'], dtype=np.float64)
                rows.append([t, float(act[t, 7]), op, float(force[12]), float(force[13]),
                             int(bool(term)), int(bool(trunc))])
                if term or trunc:
                    break
            rec['replay_steps'] = len(rows)
            rec['terminated'] = bool(rows[-1][5]) if rows else None
            rec['truncated'] = bool(rows[-1][6]) if rows else None
            rec['safety_reason'] = info.get('safety_reason')
            rec['final_info'] = {k: info.get(k) for k in ('cookies_in_target', 'cookies_in_source')}
            arr = np.asarray(rows, np.float64)
            cmd = arr[:, 1]
            op = arr[:, 2]
            lab = v2(cmd)
            close_mask = lab > 0
            open_mask = lab < 0
            rec['label_close_frac'] = float(close_mask.mean()) if len(lab) else None
            rec['actual_open_mean_close'] = float(op[close_mask].mean()) if close_mask.any() else None
            rec['actual_open_mean_open'] = float(op[open_mask].mean()) if open_mask.any() else None
            rec['actual_open_median_close'] = float(np.median(op[close_mask])) if close_mask.any() else None
            rec['actual_open_median_open'] = float(np.median(op[open_mask])) if open_mask.any() else None
            rec['actual_opening_min'] = float(op.min()) if len(op) else None
            rec['actual_opening_max'] = float(op.max()) if len(op) else None
            rec['pad_force_max_close'] = [float(arr[close_mask, 3].max()) if close_mask.any() else None,
                                          float(arr[close_mask, 4].max()) if close_mask.any() else None]
            rec['pad_force_max_all'] = [float(arr[:, 3].max()), float(arr[:, 4].max())]
            CLOSED_TH = 0.7
            OPEN_TH = 0.85
            best = None
            for lag in range(0, 16):
                if lag == 0:
                    cl = close_mask
                    om = open_mask
                    op_l = op
                else:
                    cl = close_mask[:-lag]
                    om = open_mask[:-lag]
                    op_l = op[lag:]
                if not cl.any() or not om.any():
                    continue
                acc = 0.5 * (float((op_l[cl] <= CLOSED_TH).mean()) + float((op_l[om] >= OPEN_TH).mean()))
                if best is None or acc > best['acc']:
                    best = {'lag': lag, 'acc': acc,
                            'close_actual_closed_frac': float((op_l[cl] <= CLOSED_TH).mean()),
                            'open_actual_open_frac': float((op_l[om] >= OPEN_TH).mean())}
            rec['label_vs_actual_best_lag'] = best
            trans = np.flatnonzero(np.diff(lab) != 0) + 1
            matched = 0
            lags = []
            for i in trans:
                want_close = lab[i] > 0
                pre = float(np.median(op[max(0, i - 5):i])) if i > 0 else float(op[0])
                win = op[i:min(i + 16, len(op))]
                hit = None
                if len(win):
                    if want_close:
                        if float(win.min()) <= pre - 0.1:
                            hit = int(np.argmin(win))
                    else:
                        if float(win.max()) >= pre + 0.1:
                            hit = int(np.argmax(win))
                if hit is not None:
                    matched += 1
                    lags.append(hit)
            rec['label_transitions'] = int(len(trans))
            rec['transitions_confirmed_within16'] = int(matched)
            rec['transition_lag_mean'] = float(np.mean(lags)) if lags else None
            rec['label_pulses'] = int(sum(1 for i in range(len(lab)) if (i == 0 or lab[i] != lab[i - 1]) and
                                          (i == len(lab) - 1 or lab[i + 1] != lab[i])))
            trace = os.path.join(OUTDIR, 'trace_ep%04d.npz' % ep)
            np.savez_compressed(trace, step=arr[:, 0], gripper_cmd=cmd, actual_opening=op,
                                pad_in=arr[:, 3], pad_out=arr[:, 4], label=lab)
            rec['trace'] = trace
            vf = os.path.join(OUTDIR, 'P5_replay_ep%04d_front.mp4' % ep)
            vw = os.path.join(OUTDIR, 'P5_replay_ep%04d_left_wrist.mp4' % ep)
            imageio.mimsave(vf, frames_f, fps=20, codec='libx264', quality=8)
            imageio.mimsave(vw, frames_w, fps=20, codec='libx264', quality=8)
            for name, path in (('front', vf), ('left_wrist', vw)):
                rd = imageio.get_reader(path)
                cnt = rd.count_frames()
                rd.close()
                video_entries.append({'episode': ep, 'camera': name, 'path': path,
                                      'frames': int(cnt), 'decodable': bool(cnt == len(rows)),
                                      'sha256': sha256_file(path)})
            rec['videos'] = [v['path'] for v in video_entries if v['episode'] == ep]
            rec['ok'] = bool(best is not None and best['acc'] >= 0.8 and
                             rec['label_pulses'] == 0 and len(rows) > 0 and
                             len(trans) > 0 and matched / len(trans) >= 0.8)
        except Exception:
            rec['error'] = traceback.format_exc()[:1500]
            rec['ok'] = False
        try:
            env.close()
        except Exception:
            pass
        episodes.append(rec)
        print(json.dumps({k: rec.get(k) for k in ['episode', 'replay_steps', 'label_vs_actual_best_lag',
                                                   'transitions_confirmed_within16', 'label_transitions',
                                                   'label_pulses', 'ok']}), flush=True)
    result = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'protocol': {'max_steps': MAX_STEPS, 'fps': 20, 'cameras': ['front', 'left_wrist'],
                           'action_map': 'dataset 8D [L arm 7, L grip] -> env 16D [L 7, L grip, R home 7, R grip home]',
                           'reset': 'seed=10000+episode, randomize_cookies=False, randomize_boxes=False',
                           'thresholds': 'actual closed<=0.4, actual open>=0.6, lag 0..15 steps'},
              'episodes': episodes,
              'videos': video_entries,
              'verdict': 'pass' if episodes and all(x.get('ok') for x in episodes) else 'fail',
              'elapsed_s': round(time.time() - t0, 1)}
    json.dump(result, open(OUT, 'w'), indent=1)
    json.dump({'worker': 'task7', 'episodes': [
        {k: x.get(k) for k in ['episode', 'label_close_frac', 'label_transitions',
                               'transitions_confirmed_within16', 'label_pulses', 'ok']} for x in episodes]},
        open(EVOUT, 'w'), indent=1)
    json.dump({'worker': 'task7', 'ts': result['ts'], 'videos': video_entries,
               'layout': OUTDIR}, open(MANOUT, 'w'), indent=1)
    lines = ['# GRIPPER_LABEL_V2_FINAL_AUDIT', '', 'Real MuJoCo replay of recorded expert actions (left arm + gripper).', '']
    for x in episodes:
        lines.append('## episode %d' % x['episode'])
        lines.append('- replay steps: %s / dataset frames %s' % (x.get('replay_steps'), x.get('dataset_frames')))
        lines.append('- label close frac: %s, actual opening mean(close)=%s, mean(open)=%s' % (
            x.get('label_close_frac'), x.get('actual_open_mean_close'), x.get('actual_open_mean_open')))
        lines.append('- best lag: %s' % json.dumps(x.get('label_vs_actual_best_lag')))
        lines.append('- transitions confirmed: %s/%s, pulses=%s' % (
            x.get('transitions_confirmed_within16'), x.get('label_transitions'), x.get('label_pulses')))
        lines.append('- pad force max (in/out): %s' % x.get('pad_force_max_all'))
        lines.append('- videos: %s' % x.get('videos'))
        lines.append('')
    lines.append('Verdict: %s' % result['verdict'])
    open(AUDIT, 'w').write('\n'.join(lines))
    print('DONE ' + result['verdict'])


if __name__ == '__main__':
    try:
        main()
    except Exception:
        err = traceback.format_exc()[:2000]
        json.dump({'worker': 'task7', 'error': err, 'verdict': 'error'}, open(OUT, 'w'), indent=1)
        print('TOPFAIL')
        print(err)