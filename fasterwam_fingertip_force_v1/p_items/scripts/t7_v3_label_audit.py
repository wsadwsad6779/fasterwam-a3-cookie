"""Full 749-episode audit of V3 gripper labels: open events, re-open after grip,
rapid flips (jitter), ramp smoothness, final state."""
import json
import time

import numpy as np

P = '/workspace/fwv1/p_items'
NPZ = P + '/A3_845_7D_LABELS_V3.npz'
SPLIT = P + '/DATASET_845_SPLIT_MANIFEST_V2.json'
OUT = P + '/GRIPPER_V3_LABEL_AUDIT.json'
MD = P + '/GRIPPER_V3_LABEL_AUDIT.md'


def main():
    z = np.load(NPZ)
    L = z['label'].astype(np.float64)
    ls = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    man = json.load(open(P + '/DATASET_845_ACCEPTED_MANIFEST.json'))
    accepted = {int(e['episode']) for e in man['episodes'] if e.get('accepted_for_training')}
    split = json.load(open(SPLIT))
    train_ids = set(int(x) for x in split['train_episodes'])
    rows = []
    for k in range(len(ls)):
        if k not in accepted:
            continue
        s = int(ls[k]); n = int(el[k])
        g = L[s:s + n, 6]
        if len(g) < 5:
            continue
        openm = g >= 0.8
        closed = g <= 0.6
        rises = np.flatnonzero(np.diff(openm.astype(int)) == 1) + 1
        falls = np.flatnonzero(np.diff(openm.astype(int)) == -1) + 1
        close_seg = int((np.diff(closed.astype(int)) == 1).sum())
        first_close = int(np.argmax(closed)) if closed.any() else None
        reopen = bool(openm[first_close:].any()) if first_close is not None else False
        d = np.diff(g)
        big_jumps = int((np.abs(d) > 0.25).sum())
        # rapid threshold chatter: a fall and rise (or rise+fall) within <=5 frames
        flips = 0
        ev = np.sort(np.r_[rises, falls])
        for i in range(1, len(ev)):
            if ev[i] - ev[i - 1] <= 5:
                flips += 1
        rows.append({'episode': k, 'frames': n, 'in_train': k in train_ids,
                     'starts_open': bool(g[0] >= 0.8), 'ends_open': bool(g[-1] >= 0.8),
                     'open_frac': float(openm.mean()), 'partial_frac': float(((g > 0.35) & (g < 0.8)).mean()),
                     'closed_frac': float((g <= 0.35).mean()),
                     'rise_events': int(len(rises)), 'fall_events': int(len(falls)),
                     'close_segments': close_seg, 'reopen_after_first_close': reopen,
                     'rapid_flip_pairs': int(flips), 'big_jumps_gt_0.25': big_jumps,
                     'cmd_min': float(g.min()), 'cmd_max': float(g.max()),
                     'cmd_first10': [round(float(x), 3) for x in g[:10]]})
    n = len(rows)
    out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'episodes': n,
           'starts_open': sum(1 for r in rows if r['starts_open']),
           'ends_open': sum(1 for r in rows if r['ends_open']),
           'reopen_after_first_close': sum(1 for r in rows if r['reopen_after_first_close']),
           'episodes_with_1_rise': sum(1 for r in rows if r['rise_events'] == 1),
           'episodes_with_2_plus_rises': sum(1 for r in rows if r['rise_events'] >= 2),
           'episodes_with_zero_rises': sum(1 for r in rows if r['rise_events'] == 0),
           'rapid_flip_episodes': sum(1 for r in rows if r['rapid_flip_pairs'] > 0),
           'big_jump_episodes': sum(1 for r in rows if r['big_jumps_gt_0.25'] > 0),
           'mean_open_frac': float(np.mean([r['open_frac'] for r in rows])),
           'mean_partial_frac': float(np.mean([r['partial_frac'] for r in rows])),
           'mean_closed_frac': float(np.mean([r['closed_frac'] for r in rows])),
           'no_reopen_episodes': [r['episode'] for r in rows if not r['reopen_after_first_close']][:50],
           'zero_rise_episodes': [r['episode'] for r in rows if r['rise_events'] == 0][:50],
           'rapid_flip_episode_sample': [{'episode': r['episode'], 'pairs': r['rapid_flip_pairs']}
                                          for r in rows if r['rapid_flip_pairs'] > 0][:20],
           'checks': {}}
    c = out['checks']
    c['all_start_open'] = out['starts_open'] == n
    c['all_reopen_after_grip'] = out['reopen_after_first_close'] == n
    c['no_rapid_threshold_chatter'] = out['rapid_flip_episodes'] == 0
    c['no_large_command_jumps'] = out['big_jump_episodes'] == 0
    c['has_two_close_segments_or_more'] = all(r['close_segments'] >= 2 for r in rows) if rows else False
    out['verdict'] = 'pass' if all(c.values()) else 'fail'
    json.dump(out, open(OUT, 'w'), indent=1)
    md = ['# GRIPPER V3 LABEL AUDIT (749 accepted episodes)', '',
          '- starts_open %d/%d, reopens after first grip %d/%d' % (out['starts_open'], n, out['reopen_after_first_close'], n),
          '- episodes with >=2 close segments: %d/%d' % (sum(1 for r in rows if r['close_segments'] >= 2), n),
          '- rises: 0-rise %d, 1-rise %d, >=2-rise %d' % (out['episodes_with_zero_rises'], out['episodes_with_1_rise'], out['episodes_with_2_plus_rises']),
          '- rapid threshold chatter episodes: %d; large command jumps (>0.25/frame): %d' % (out['rapid_flip_episodes'], out['big_jump_episodes']),
          '- frame fractions: open %.3f / partial %.3f / closed %.3f' % (out['mean_open_frac'], out['mean_partial_frac'], out['mean_closed_frac']),
          '- verdict: **%s**' % out['verdict'], '',
          'Checks: %s' % json.dumps(c)]
    open(MD, 'w').write('\n'.join(md))
    print(json.dumps({k: out[k] for k in ['episodes', 'starts_open', 'reopen_after_first_close',
                                          'episodes_with_zero_rises', 'rapid_flip_episodes',
                                          'big_jump_episodes', 'mean_open_frac', 'verdict']}))


if __name__ == '__main__':
    main()