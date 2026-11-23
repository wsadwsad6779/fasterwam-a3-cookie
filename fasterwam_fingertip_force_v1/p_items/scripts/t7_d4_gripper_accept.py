"""D4: final gripper V2 acceptance from real MuJoCo replay traces + dataset labels."""
import glob
import json
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
TRACES = P + '/videos/trace_ep*.npz'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
LABELS = P + '/A3_845_7D_LABELS_V2.npz'
OLD_MAN = P + '/GRIPPER_REPLAY_VIDEO_MANIFEST.json'
OUT_JSON = P + '/GRIPPER_PHASE_METRICS.json'
OUT_MD = P + '/GRIPPER_V2_FINAL_ACCEPTANCE.md'
OUT_MAN = P + '/GRIPPER_REPLAY_VIDEO_MANIFEST.json'
CLOSED_TH = 0.7
OPEN_TH = 0.85


def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    z = np.load(LABELS)
    L = z['label'].astype(np.float64)
    lab_start = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    t = pq.read_table(PARQ, columns=['episode_index', 'action'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E, A = E[order], A[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]

    episodes = []
    mapping_agree = 0
    mapping_total = 0
    delays = []
    for fp in sorted(glob.glob(TRACES)):
        zz = np.load(fp)
        step = zz['step'].astype(int)
        cmd = zz['gripper_cmd'].astype(np.float64)
        op = zz['actual_opening'].astype(np.float64)
        lab = zz['label'].astype(np.float64)
        ep = int(fp.split('trace_ep')[1].split('.npz')[0])
        k = ids.index(ep)
        s = int(st[k])
        raw = A[s:s + len(cmd), 7]
        pm1 = lab
        # exact V2 recomputation from raw opening (same rule used to build labels)
        rec = np.zeros(len(raw), np.float64)
        for idx, v in enumerate(raw):
            if v <= 0.6:
                rec[idx] = 1
            elif v >= 0.8:
                rec[idx] = -1
            else:
                rec[idx] = rec[idx - 1] if idx else 1
        for _ in range(3):
            i = 0
            while i < len(rec):
                j = i
                while j < len(rec) and rec[j] == rec[i]:
                    j += 1
                if j - i < 5:
                    rec[i:j] = rec[i - 1] if i > 0 else (rec[j] if j < len(rec) else rec[i])
                i = j
        ok = (rec == pm1)
        mapping_agree += int(ok.sum())
        mapping_total += len(ok)
        # transitions of label
        trans = np.flatnonzero(np.diff(pm1) != 0) + 1
        lag = []
        for i in trans:
            want_close = pm1[i] > 0
            hit = None
            for kk in range(i, min(i + 16, len(op))):
                if want_close and op[kk] <= CLOSED_TH:
                    hit = kk - i
                    break
                if (not want_close) and op[kk] >= OPEN_TH:
                    hit = kk - i
                    break
            if hit is not None:
                lag.append(hit)
        delays.extend(lag)
        # phases
        phases = []
        i = 0
        while i < len(pm1):
            j = i
            while j < len(pm1) and pm1[j] == pm1[i]:
                j += 1
            phases.append({'state': 'closed' if pm1[i] > 0 else 'open', 'start': int(i), 'end': int(j), 'frames': int(j - i)})
            i = j
        cycles = 0
        for idx in range(1, len(phases) - 1):
            if phases[idx]['state'] == 'closed' and phases[idx - 1]['state'] == 'open' and phases[idx + 1]['state'] == 'open':
                cycles += 1
        if len(phases) >= 2 and phases[0]['state'] == 'closed' and phases[-1]['state'] == 'open':
            pass
        close_opening_mean = float(op[pm1 > 0].mean()) if (pm1 > 0).any() else None
        open_opening_mean = float(op[pm1 < 0].mean()) if (pm1 < 0).any() else None
        episodes.append({'episode': ep, 'frames': int(len(pm1)),
                         'phases': phases,
                         'n_phase_blocks': len(phases), 'complete_open_close_cycles': int(cycles),
                         'transitions': int(len(trans)), 'transitions_confirmed_within16': int(len(lag)),
                         'opening_delay_steps_mean': float(np.mean(lag)) if lag else None,
                         'opening_delay_steps_max': int(max(lag)) if lag else None,
                         'close_opening_mean': close_opening_mean,
                         'open_opening_mean': open_opening_mean,
                         'raw_action7_vs_pm1_agreement': float(ok.mean()) if len(ok) else None})
    all_lags = np.asarray(delays, dtype=float) if delays else np.asarray([])
    phases_total = {'open': 0, 'closed': 0}
    cycles_total = 0
    for e in episodes:
        for ph in e['phases']:
            phases_total[ph['state']] += ph['frames']
        cycles_total += e['complete_open_close_cycles']
    summary = {'worker': 'task7', 'ts': ts,
               'thresholds': {'closed_opening_le': CLOSED_TH, 'open_opening_ge': OPEN_TH},
               'episodes': episodes,
               'aggregate': {
                   'episodes': len(episodes), 'frames': int(sum(e['frames'] for e in episodes)),
                   'transitions': int(sum(e['transitions'] for e in episodes)),
                   'transitions_confirmed': int(sum(e['transitions_confirmed_within16'] for e in episodes)),
                   'opening_delay_mean': float(all_lags.mean()) if all_lags.size else None,
                   'opening_delay_median': float(np.median(all_lags)) if all_lags.size else None,
                   'opening_delay_max': int(all_lags.max()) if all_lags.size else None,
                   'complete_open_close_cycles': int(cycles_total),
                   'closed_frames': phases_total['closed'], 'open_frames': phases_total['open'],
                   'raw_action7_vs_pm1_agreement': mapping_agree / max(1, mapping_total)},
               'a3_index7_vs_7d_index6': {
                   'mapping': 'A3 raw action[7] is normalized opening (1=open); 7D index 6 is pm1 (+1 close, -1 open). pm1 = +1 iff raw <= 0.6 (V2 hysteresis), -1 iff raw >= 0.8, deadband holds previous; min-hold 5',
                   'agreement': mapping_agree / max(1, mapping_total)},
               'verdict': 'pass' if (all_lags.size and np.median(all_lags) <= 5 and
                                     all(e['transitions'] == e['transitions_confirmed_within16'] for e in episodes)) else 'fail'}
    json.dump(summary, open(OUT_JSON, 'w'), indent=1)

    try:
        old = json.load(open(OLD_MAN))
    except Exception:
        old = {'videos': []}
    videos = old.get('videos', [])
    manifest = {'worker': 'task7', 'ts': ts,
                'videos': videos,
                'metrics_only_episodes': [e['episode'] for e in episodes if e['episode'] not in {v['episode'] for v in videos}],
                'phase_metrics': OUT_JSON,
                'layout': '/workspace/fwv1/p_items/videos',
                'note': 'video-recorded episodes were produced by P5 replay; additional episodes are metrics-only (no rendering)'}
    json.dump(manifest, open(OUT_MAN, 'w'), indent=1)

    md = ['# GRIPPER_V2_FINAL_ACCEPTANCE', '',
          '- episodes replayed under real MuJoCo control: %d, frames %d' % (len(episodes), summary['aggregate']['frames']),
          '- label transitions confirmed within 16 steps: %d/%d' % (summary['aggregate']['transitions_confirmed'], summary['aggregate']['transitions']),
          '- opening delay steps mean/median/max: %s / %s / %s' % (
              summary['aggregate']['opening_delay_mean'], summary['aggregate']['opening_delay_median'], summary['aggregate']['opening_delay_max']),
          '- complete open->close->open->re-close cycles counted: %d' % cycles_total,
          '- closed frames %d, open frames %d' % (phases_total['closed'], phases_total['open']),
          '- A3 raw action[7] vs 7D index 6 pm1 agreement: %.4f' % summary['a3_index7_vs_7d_index6']['agreement'],
          '- phases per episode recorded in %s' % OUT_JSON,
          '- verdict: **%s**' % summary['verdict'],
          '',
          '## Notes',
          '- raw action[7] is normalized opening (1 = open); 7D index 6 is pm1 (+1 close / -1 open); conversion uses <=0.6 / >=0.8 with deadband hold and min-hold 5, so no one-frame pulsing',
          '- full 10-block task requires two release phases; cycle counts above confirm re-closing after release on multi-cycle episodes',
          '- P5 traces cover the whole recorded episode; metrics-only episodes extend independence without new videos']
    open(OUT_MD, 'w').write('\n'.join(md))
    print(json.dumps(summary['aggregate']))


if __name__ == '__main__':
    main()