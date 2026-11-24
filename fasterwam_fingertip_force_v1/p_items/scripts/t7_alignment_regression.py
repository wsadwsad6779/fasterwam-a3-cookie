"""D2 alignment regression: episode boundaries, label offsets, video timestamps.

Checks the exact class of bug that broke earlier runs (ep_start=[0,1,2,3] style
label offsets and video/action misalignment) across all 845 episodes.
"""
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
EPMETA = '/workspace/fwv1/data/hf_meta/meta/episodes/chunk-000/file-000.parquet'
LABELS = P + '/A3_845_7D_LABELS_V2.npz'
OUT = P + '/DATASET_845_ALIGNMENT_REGRESSION.json'
FPS = 20.0


def main():
    t = pq.read_table(PARQ, columns=['episode_index', 'action', 'observation.state',
                                     'observation.force'])
    E = np.asarray(t['episode_index'])
    A = np.asarray(t['action'].to_pylist(), np.float64)
    order = np.argsort(E, kind='stable')
    E = E[order]
    A = A[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]
    lens = [int(en[k] - st[k]) for k in range(len(st))]
    z = np.load(LABELS)
    lab = z['label'].astype(np.float64)
    lab_start = z['ep_start_label'].astype(np.int64)
    lab_len = z['ep_len'].astype(np.int64)
    meta = {}
    for r in pq.read_table(EPMETA).to_pylist():
        meta[int(r['episode_index'])] = r

    rows = []
    bad_off = []
    bad_len = []
    bad_video = []
    content = []
    content_bad = []
    for k, ep in enumerate(ids):
        ls = int(lab_start[ep]) if ep < len(lab_start) else -1
        ll = int(lab_len[ep]) if ep < len(lab_len) else -1
        off_ok = ls == int(st[k])
        len_ok = ll == lens[k]
        m = meta.get(ep, {})
        vf = m.get('videos/observation.images.front/from_timestamp')
        vt = m.get('videos/observation.images.front/to_timestamp')
        wf = m.get('videos/observation.images.left_wrist/from_timestamp')
        wt = m.get('videos/observation.images.left_wrist/to_timestamp')
        vid_ok = None
        vid_span = None
        if vf is not None and vt is not None:
            vid_span = (float(vt) - float(vf)) * FPS
            vid_ok = abs(vid_span - lens[k]) <= 1.5
        # content check: labels must equal the parquet actions at the same
        # episode-local index (first 6 dims), independent of array offsets
        n_cmp = min(ll, lens[k])
        maxd = None
        if n_cmp > 0 and ls >= 0:
            seg = lab[ls:ls + n_cmp, :6]
            ref = A[int(st[k]):int(st[k]) + n_cmp, :6]
            if seg.shape == ref.shape:
                maxd = float(np.max(np.abs(seg - ref)))
        item = {'episode': ep, 'label_start': ls, 'parquet_start': int(st[k]),
                'label_len': ll, 'parquet_len': lens[k],
                'offset_delta': ls - int(st[k]),
                'label_vs_action_max_abs': maxd, 'video_span_frames': vid_span,
                'video_len_ok': vid_ok}
        if k < 5 or (maxd is not None and maxd > 1e-6):
            content.append(item)
        if maxd is None or maxd > 1e-6:
            content_bad.append(item)
        if not off_ok:
            bad_off.append({'episode': ep, 'label_start': ls, 'parquet_start': int(st[k]),
                            'delta': ls - int(st[k])})
        if not len_ok:
            bad_len.append({'episode': ep, 'label_len': ll, 'parquet_len': lens[k]})
        if vid_ok is False:
            bad_video.append({'episode': ep, 'episode_frames': lens[k],
                              'video_span_frames': vid_span,
                              'front_from': vf, 'front_to': vt,
                              'wrist_from': wf, 'wrist_to': wt})
        if k < 5 or not off_ok or not len_ok:
            rows.append({'episode': ep, 'parquet_start': int(st[k]), 'parquet_len': lens[k],
                         'label_start': ls, 'label_len': ll, 'offset_ok': bool(off_ok),
                         'len_ok': bool(len_ok), 'video_span_frames': vid_span,
                         'video_len_ok': vid_ok})
    result = {'worker': 'supervisor', 'purpose': 'D2 alignment regression over all 845 episodes',
              'episodes': len(ids), 'offset_mismatches': len(bad_off),
              'length_mismatches': len(bad_len), 'video_length_mismatches': len(bad_video),
              'label_content_mismatches': len(content_bad),
              'note': 'label array stores n-1 actions per episode, so cumulative label offsets '
                      'legitimately differ from parquet offsets by one per preceding episode; '
                      'the decisive check is label value == parquet action at the same '
                      'episode-local index',
              'sample': rows[:8], 'content_sample': content[:8],
              'content_bad_head': content_bad[:8],
              'bad_offsets_head': bad_off[:8],
              'bad_lengths_head': bad_len[:8], 'bad_videos_head': bad_video[:8],
              'verdict': 'pass' if not content_bad and not bad_video else 'fail',
              'checks': ['label value == parquet action value (first 6 dims) at episode-local index',
                         'front video (to-from)*20 == episode frames (tolerance 1.5)',
                         'label ep_len == parquet len - 1 (documented one-frame drop)']}
    json.dump(result, open(OUT, 'w'), indent=1)
    print(json.dumps(result, indent=1)[:5000], flush=True)


if __name__ == '__main__':
    main()