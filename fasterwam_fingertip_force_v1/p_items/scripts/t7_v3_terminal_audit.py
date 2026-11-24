"""Check whether episodes actually end with a release (opening) and how labels
represent the final grasp-release cycle."""
import json
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
V3 = P + '/A3_845_7D_LABELS_V3.npz'
MAN = P + '/DATASET_845_ACCEPTED_MANIFEST.json'
OUT = P + '/GRIPPER_V3_TERMINAL_STATE_AUDIT.json'
MD = P + '/GRIPPER_V3_TERMINAL_STATE_AUDIT.md'


def main():
    man = json.load(open(MAN))
    accepted = {int(e['episode']) for e in man['episodes'] if e.get('accepted_for_training')}
    t = pq.read_table(PARQ, columns=['episode_index', 'action'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E, A = E[order], A[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]
    z = np.load(V3)
    L = z['label']
    ls = z['ep_start_label'].astype(np.int64)
    rows = []
    for k, ep in enumerate(ids):
        if ep not in accepted:
            continue
        s, e = int(st[k]), int(en[k])
        g = A[s:e, 7]
        last = g[-1]
        last10 = [round(float(x), 3) for x in g[-10:]]
        first = float(g[0])
        lab_last = float(L[int(ls[k]) + int(e - s) - 2, 6])
        # final-rise: does the raw command rise above 0.8 within the last 40 frames?
        final_rise = bool((g[-40:] >= 0.8).sum() > 0) and bool(last >= 0.8)
        rows.append({'episode': ep, 'frames': int(e - s), 'first_raw': round(first, 3),
                     'last_raw': round(last, 3), 'label_last': round(lab_last, 3),
                     'last10_raw': last10,
                     'ends_open_raw': bool(last >= 0.8),
                     'ends_closed_raw': bool(last <= 0.6),
                     'final_rise_to_open': final_rise})
    n = len(rows)
    out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'episodes': n,
           'first_open_ge0.8': sum(1 for r in rows if r['first_raw'] >= 0.8),
           'last_open_ge0.8': sum(1 for r in rows if r['ends_open_raw']),
           'last_closed_le0.6': sum(1 for r in rows if r['ends_closed_raw']),
           'final_rise_to_open': sum(1 for r in rows if r['final_rise_to_open']),
           'label_last_open': sum(1 for r in rows if r['label_last'] >= 0.8),
           'sample_last10': [{'episode': r['episode'], 'last10': r['last10_raw'],
                              'last_raw': r['last_raw'], 'label_last': r['label_last']}
                             for r in rows[:5]],
           'verdict': 'pass_with_terminal_release' if sum(1 for r in rows if r['ends_open_raw']) == n
                      else 'incomplete_terminal_release'}
    json.dump(out, open(OUT, 'w'), indent=1)
    md = ['# GRIPPER V3 TERMINAL STATE AUDIT', '',
          '- episodes: %d' % n,
          '- first frame open (>=0.8): %d' % out['first_open_ge0.8'],
          '- final frame open (>=0.8): %d' % out['last_open_ge0.8'],
          '- final frame closed (<=0.6): %d' % out['last_closed_le0.6'],
          '- final rise to open in last 40 frames: %d' % out['final_rise_to_open'],
          '- label last row open (>=0.8): %d' % out['label_last_open'],
          '- verdict: **%s**' % out['verdict'], '',
          'If final-open is 0, the recorded episodes end with the gripper still closed; the mid-episode release is present, but the final release after the last batch is not demonstrated in the data and must be handled before claiming a complete open/close cycle.']
    open(MD, 'w').write('\n'.join(md))
    print(json.dumps({k: out[k] for k in ['episodes', 'first_open_ge0.8', 'last_open_ge0.8',
                                          'last_closed_le0.6', 'final_rise_to_open', 'verdict']}))


if __name__ == '__main__':
    main()