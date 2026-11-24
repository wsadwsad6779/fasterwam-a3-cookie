"""Fingertip contact force by gripper command level.

The V3 gripper label is the raw commanded opening (1=open). This script bins
dataset frames by that command level and reports the left fingertip pad contact
force (channels 12/13, in N) so that the physical meaning of each level is
established from the data:

  grasp band   cmd<=0.35  (the level used while carrying a cookie)
  release band 0.35<cmd<0.70
  open band    cmd>=0.8

If the grasp band carries sustained pad force, the level is a real clamp on the
cookie (cookie is 21.5 mm thick on the grip axis, 2F-85 travel is 85 mm, so the
free-space jaw gap at cmd 0.28 would be ~24 mm). If it does not, the level is
just a jaw position and the grasp must be explained by something else.
"""
import json
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
MAN = P + '/DATASET_845_ACCEPTED_MANIFEST.json'
OUT = P + '/GRIPPER_V3_FORCE_BY_LEVEL.json'
MD = P + '/GRIPPER_V3_FORCE_BY_LEVEL.md'


def main():
    man = json.load(open(MAN))
    accepted = {int(e['episode']) for e in man['episodes'] if e.get('accepted_for_training')}
    t = pq.read_table(PARQ, columns=['episode_index', 'action', 'observation.force'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    F = np.asarray(t['observation.force'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E, A, F = E[order], A[order], F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]

    def stats(v):
        if len(v) == 0:
            return None
        return {'n': int(len(v)), 'mean': float(v.mean()), 'median': float(np.median(v)),
                'p90': float(np.percentile(v, 90)),
                'frac_gt_0p2': float((v > 0.2).mean()),
                'frac_gt_1': float((v > 1.0).mean())}

    rows = []
    for k, ep in enumerate(ids):
        if ep not in accepted:
            continue
        s, e = int(st[k]), int(en[k])
        cmd = A[s:e, 7]
        pad = F[s:e, 12:14].max(axis=1)
        row = {'episode': ep}
        for name, m in (('grasp', cmd <= 0.35), ('release', (cmd > 0.35) & (cmd < 0.70)),
                        ('open', cmd >= 0.8)):
            v = pad[m]
            row[name] = stats(v)
        # steady middle of grasp runs (skip 10 frames after entry)
        steady = np.zeros_like(cmd, dtype=bool)
        i = 0
        n = len(cmd)
        while i < n:
            if cmd[i] <= 0.35:
                j = i
                while j < n and cmd[j] <= 0.35:
                    j += 1
                if j - i > 20:
                    steady[i + 10:j] = True
                i = j
            else:
                i += 1
        row['grasp_steady'] = stats(pad[steady])
        row['right_pad_nonzero_frac'] = float((F[s:e, 14:16] > 1e-9).mean())
        rows.append(row)

    def agg(band):
        vals = [r[band]['median'] for r in rows if r[band]]
        means = [r[band]['mean'] for r in rows if r[band]]
        return {'episodes': len(vals),
                'median_of_episode_medians': float(np.median(vals)) if vals else None,
                'median_of_episode_means': float(np.median(means)) if means else None,
                'episode_frac_force_gt_0p2': float(np.mean([r[band]['frac_gt_0p2'] for r in rows if r[band]])) if vals else None,
                'episode_frac_force_gt_1': float(np.mean([r[band]['frac_gt_1'] for r in rows if r[band]])) if vals else None}

    out = {
        'worker': 'task7',
        'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'source': 'action[7] command level vs observation.force[12:14] (left pad, N)',
        'episodes': len(rows),
        'grasp': agg('grasp'), 'grasp_steady': agg('grasp_steady'),
        'release': agg('release'), 'open': agg('open'),
        'right_pad_nonzero_frac_mean': float(np.mean([r['right_pad_nonzero_frac'] for r in rows])),
        'sample': [{k: r[k] for k in ('episode', 'grasp', 'grasp_steady', 'release', 'open')} for r in rows[:3]],
    }
    g = out['grasp_steady']
    verdict = ('grasp_is_clamp' if g['median_of_episode_medians'] is not None and g['median_of_episode_medians'] > 0.5
               and g['episode_frac_force_gt_0p2'] > 0.5 else 'grasp_not_clamp')
    out['verdict'] = verdict
    json.dump(out, open(OUT, 'w'), indent=1)
    md = ['# GRIPPER V3 FORCE BY COMMAND LEVEL', '',
          '- episodes: %d' % len(rows),
          '- grasp (cmd<=0.35) steady: median pad force %.3f N, fraction>0.2N %.3f' % (
              g['median_of_episode_medians'], g['episode_frac_force_gt_0p2']),
          '- release (0.35<cmd<0.7): median pad force %.3f N, fraction>0.2N %.3f' % (
              out['release']['median_of_episode_medians'], out['release']['episode_frac_force_gt_0p2']),
          '- open (cmd>=0.8): median pad force %.3f N' % out['open']['median_of_episode_medians'],
          '- right pad nonzero fraction: %.5f' % out['right_pad_nonzero_frac_mean'],
          '- verdict: **%s**' % verdict, '',
          'Cookie grip-axis thickness is 0.0215 m and 2F-85 travel is 0.085 m, so a free',
          'jaw at cmd 0.28 sits near 0.024 m; sustained pad force at the grasp level shows',
          'the fingers are pressing the cookie (a real clamp), not hovering at a position.']
    open(MD, 'w').write('\n'.join(md))
    print(json.dumps({k: out[k] for k in ('episodes', 'grasp', 'grasp_steady', 'release', 'open',
                                          'right_pad_nonzero_frac_mean', 'verdict')}))


if __name__ == '__main__':
    main()