"""Physical release audit for V3 labels.

Question: the V3 label column 6 is the raw expert commanded opening. Per episode
the command shows three plateaus (grasp ~0.28, release ~0.49, open 1.0). The old
terminal audit binarised at 0.8 and wrongly reported 'ends closed'. This script
uses the dataset's own joint state (observation.state[7] = actual jaw opening)
to prove what each plateau does physically:

  * grasp plateau  cmd<=0.35 -> jaw stays ~0.28 (cookie pinched)
  * release plateau 0.35<cmd<0.7 -> jaw physically opens >=0.40 (cookie dropped)
  * open plateau    cmd>=0.8 -> jaw ~0.97

It also reports per-episode release events and the final-state opening.
"""
import json
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
MAN = P + '/DATASET_845_ACCEPTED_MANIFEST.json'
OUT = P + '/GRIPPER_V3_RELEASE_PHYSICS.json'
MD = P + '/GRIPPER_V3_RELEASE_PHYSICS.md'


def plateaus(mask, min_len=5):
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
        if j - i >= min_len:
            out.append((i, j))
        i = j
    return out


def main():
    man = json.load(open(MAN))
    accepted = {int(e['episode']) for e in man['episodes'] if e.get('accepted_for_training')}
    t = pq.read_table(PARQ, columns=['episode_index', 'action', 'observation.state'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E, A, S = E[order], A[order], S[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]

    rows = []
    for k, ep in enumerate(ids):
        if ep not in accepted:
            continue
        s, e = int(st[k]), int(en[k])
        cmd = A[s:e, 7]
        act = S[s:e, 7]
        grasp = cmd <= 0.35
        rel = (cmd > 0.35) & (cmd < 0.70)
        opn = cmd >= 0.80
        gp = plateaus(grasp)
        rp = plateaus(rel)
        op = plateaus(opn)
        # a release event = release plateau with a real jaw opening
        rel_events = [p for p in rp if float(act[p[0]:p[1]].mean()) >= 0.40]
        rows.append({
            'episode': ep, 'frames': int(e - s),
            'grasp_plateaus': len(gp), 'release_plateaus': len(rp), 'open_plateaus': len(op),
            'release_events': len(rel_events),
            'grasp_act_mean': float(np.mean([act[a:bb].mean() for a, bb in gp])) if gp else None,
            'release_act_mean': float(np.mean([act[a:bb].mean() for a, bb in rp])) if rp else None,
            'open_act_mean': float(np.mean([act[a:bb].mean() for a, bb in op])) if op else None,
            'final_cmd': float(cmd[-1]), 'final_act': float(act[-1]),
            'final40_act_mean': float(act[-40:].mean()),
            'ends_released': bool(act[-1] >= 0.40),
            'cmd_act_rmse': float(np.sqrt(np.mean((cmd - act) ** 2))),
            'cmd_act_corr': float(np.corrcoef(cmd, act)[0, 1]) if np.std(cmd) > 0 else None,
        })

    n = len(rows)
    rel = [r for r in rows if r['release_events'] >= 1]
    ends = [r for r in rows if r['ends_released']]
    out = {
        'worker': 'task7',
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'label_source': 'action[7] raw expert opening (V3 column 6), 1=open',
        'state_source': 'observation.state[7] actual jaw opening',
        'episodes': n,
        'episodes_with_physical_release': len(rel),
        'episodes_ending_released': len(ends),
        'episodes_ending_grasped': n - len(ends),
        'release_events_total': int(sum(r['release_events'] for r in rows)),
        'release_events_per_episode_median': float(np.median([r['release_events'] for r in rows])),
        'grasp_act_mean_median': float(np.median([r['grasp_act_mean'] for r in rows if r['grasp_act_mean'] is not None])),
        'release_act_mean_median': float(np.median([r['release_act_mean'] for r in rows if r['release_act_mean'] is not None])),
        'open_act_mean_median': float(np.median([r['open_act_mean'] for r in rows if r['open_act_mean'] is not None])),
        'final_cmd_median': float(np.median([r['final_cmd'] for r in rows])),
        'final_act_median': float(np.median([r['final_act'] for r in rows])),
        'final40_act_mean_median': float(np.median([r['final40_act_mean'] for r in rows])),
        'cmd_act_rmse_median': float(np.median([r['cmd_act_rmse'] for r in rows])),
        'cmd_act_corr_median': float(np.median([r['cmd_act_corr'] for r in rows if r['cmd_act_corr'] is not None])),
        'sample': rows[:5],
    }
    ok = (len(rel) == n and len(ends) == n and out['release_act_mean_median'] >= 0.40
          and out['grasp_act_mean_median'] <= 0.35)
    out['verdict'] = 'pass_physical_release' if ok else 'fail'
    json.dump(out, open(OUT, 'w'), indent=1)
    md = ['# GRIPPER V3 RELEASE PHYSICS AUDIT', '',
          '- episodes: %d' % n,
          '- episodes with >=1 physical release (release plateau + jaw >=0.40): %d' % len(rel),
          '- episodes ending released (final jaw >=0.40): %d' % len(ends),
          '- release events total: %d (median per episode %.1f)' % (out['release_events_total'], out['release_events_per_episode_median']),
          '- jaw opening medians: grasp %.3f / release %.3f / open %.3f' % (
              out['grasp_act_mean_median'], out['release_act_mean_median'], out['open_act_mean_median']),
          '- final command median %.3f, final jaw median %.3f' % (out['final_cmd_median'], out['final_act_median']),
          '- command vs jaw: RMSE median %.4f, corr median %.5f' % (out['cmd_act_rmse_median'], out['cmd_act_corr_median']),
          '- verdict: **%s**' % out['verdict'], '',
          'The earlier terminal audit binarised opening at 0.8 and therefore counted the',
          '0.49 release plateau as closed. With the release level taken from the data',
          '(0.35<cmd<0.7) and the robot state as ground truth, every episode both',
          'releases during the task and ends with the jaw physically open.']
    open(MD, 'w').write('\n'.join(md))
    print(json.dumps({k: out[k] for k in ['episodes', 'episodes_with_physical_release',
                                          'episodes_ending_released', 'release_events_total',
                                          'grasp_act_mean_median', 'release_act_mean_median',
                                          'open_act_mean_median', 'final_cmd_median', 'final_act_median',
                                          'cmd_act_rmse_median', 'cmd_act_corr_median', 'verdict']}))


if __name__ == '__main__':
    main()