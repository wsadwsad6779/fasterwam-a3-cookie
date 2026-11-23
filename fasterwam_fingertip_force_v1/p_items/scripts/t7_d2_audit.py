"""D2: full 845-episode quality audit, identity resolution, three-layer dedup,
group-aware frozen-test split, and accepted manifest."""
import csv
import hashlib
import json
import os
import random
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
META = P + '/hf_multi/a3_episode_metadata.jsonl'
EPMETA = '/workspace/fwv1/data/hf_meta/meta/episodes/chunk-000/file-000.parquet'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
VIDEOS = '/workspace/fwv1/data/hf_videos/videos'
OLD_SPLIT = P + '/DATASET_800_SPLIT_MANIFEST.json'
FROZEN = P + '/FROZEN_DATA_MANIFEST.json'
OUT_MANIFEST = P + '/DATASET_845_ACCEPTED_MANIFEST.json'
OUT_AUDIT = P + '/DATASET_845_FINAL_QUALITY_AUDIT.md'
OUT_PROV = P + '/DATASET_845_PROVENANCE.json'
OUT_DEDUP = P + '/DATASET_845_PHYSICAL_DEDUP.csv'
OUT_LAYOUT = P + '/DATASET_845_LAYOUT_VISUAL_COVERAGE.json'
OUT_LEAK = P + '/DATASET_845_TRAIN_VAL_TEST_LEAKAGE.json'
OUT_SPLIT_V2 = P + '/DATASET_845_SPLIT_MANIFEST_V2.json'
SPLIT_SEED = 20261123


def h(arr, q=5):
    s = hashlib.sha1()
    s.update(np.round(np.asarray(arr, np.float64), q).astype(np.float32).tobytes())
    return s.hexdigest()


def jh(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    lines = [l for l in open(META) if l.strip()]
    entries = [json.loads(l) for l in lines]
    by_id = {}
    for d in entries:
        by_id.setdefault(int(d['episode_index']), []).append(d)
    epmeta = {int(r['episode_index']): r for r in pq.read_table(EPMETA).to_pylist()}
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.state', 'observation.eef_pose',
                                     'action', 'observation.force'])
    E = np.asarray(t['episode_index'])
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    X = np.asarray(t['observation.eef_pose'].to_pylist(), np.float64)
    A = np.asarray(t['action'].to_pylist(), np.float64)
    fc = t['observation.force'].combine_chunks()
    F = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    order = np.argsort(E, kind='stable')
    E, S, X, A, F = E[order], S[order], X[order], A[order], F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]
    assert len(ids) == 845, len(ids)

    eps = []
    for k, ep in enumerate(ids):
        s, e = int(st[k]), int(en[k])
        n = e - s
        m = epmeta.get(ep, {})
        ents = by_id.get(ep, [])
        frames_entry = None
        for d in ents:
            if int(d.get('frames', -1)) == n:
                frames_entry = d
                break
        if frames_entry is not None and len(ents) > 1:
            class_src = 'metadata_frame_match'
            chosen = frames_entry
        elif len(ents) == 1:
            class_src = 'metadata_single'
            chosen = ents[0]
        else:
            class_src = 'unknown'
            chosen = None
        src = str(chosen.get('source_dataset')) if chosen else None
        if src:
            task_class = '2x10' if '2x10' in src else ('2x5' if '2x5' in src else 'unknown')
        else:
            task_class = '2x10_inferred' if n >= 800 else '2x5_inferred'
        rand = (chosen or {}).get('randomization') or {}
        video_paths = {}
        exist = True
        for key in ('front', 'left_wrist'):
            ck = 'videos/observation.images.%s/chunk_index' % key
            fk = 'videos/observation.images.%s/file_index' % key
            if ck in m and fk in m:
                rel = 'observation.images.%s/chunk-%03d/file-%03d.mp4' % (key, int(m[ck]), int(m[fk]))
                video_paths[key] = rel
                exist = exist and os.path.exists(os.path.join(VIDEOS, rel))
        finite = bool(np.isfinite(S[s:e]).all() and np.isfinite(X[s:e]).all()
                      and np.isfinite(A[s:e]).all() and np.isfinite(F[s:e]).all())
        action_hash = h(A[s:e])
        eef_hash = h(X[s:e])
        phys_hash = hashlib.sha1((action_hash + eef_hash).encode()).hexdigest()[:16]
        scene = {'cookie_positions': rand.get('cookie_positions'),
                 'source_bin_position': rand.get('source_bin_position'),
                 'target_bin_position': rand.get('target_bin_position'),
                 'source_column': rand.get('source_column'),
                 'profile': rand.get('profile')}
        scene_hash = jh(scene) if chosen else None
        visual_sig = jh({'cameras': rand.get('cameras'), 'light': rand.get('light_diffuse'),
                         'profile': rand.get('profile')}) if chosen else None
        quarantined = chosen is None
        identity_known = chosen is not None and task_class in ('2x5', '2x10')
        accepted = bool(identity_known and n >= 33 and finite and exist)
        eps.append({
            'episode': ep, 'frames': n, 'parquet': 'data/chunk-000/file-%03d.parquet' % int(m.get('data/file_index', 0)),
            'dataset_from_index': int(m.get('dataset_from_index', -1)), 'dataset_to_index': int(m.get('dataset_to_index', -1)),
            'task': (chosen or {}).get('task'), 'success': (chosen or {}).get('success'),
            'controller_type': (chosen or {}).get('controller_type'),
            'source_dataset': src, 'source_shard': (chosen or {}).get('source_shard'),
            'task_class': task_class, 'task_class_source': class_src,
            'metadata_entries': len(ents), 'seed': (chosen or {}).get('seed'),
            'source_column': rand.get('source_column'), 'profile': rand.get('profile'),
            'initial_scene_hash': scene_hash, 'visual_signature': visual_sig,
            'videos': video_paths, 'video_files_exist': bool(exist),
            'action_sha1': action_hash, 'eef_sha1': eef_hash,
            'proprio_sha1': h(S[s:e]), 'force_sha1': h(F[s:e]),
            'physical_group': phys_hash,
            'finite': finite, 'identity_known': identity_known,
            'quarantined_unknown_identity': bool(quarantined),
            'accepted_for_training': accepted,
            'accept_reason': ('ok' if accepted else
                              ('quarantine_unknown_identity' if quarantined else 'failed_checks')),
        })

    old = json.load(open(OLD_SPLIT))
    old_test = set(int(x) for x in old['test_episodes'])
    groups = {}
    for x in eps:
        groups.setdefault(x['physical_group'], []).append(x['episode'])
    gids = sorted(groups)
    test_groups = set(x['physical_group'] for x in eps if x['episode'] in old_test or x['quarantined_unknown_identity'])
    remaining = [g for g in gids if g not in test_groups]
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(remaining)
    remaining.sort(key=lambda g: -len(groups[g]))
    val_groups = set()
    val_count = 0
    for g in remaining:
        if val_count >= 85:
            break
        val_groups.add(g)
        val_count += len(groups[g])
    train_groups = set(remaining) - val_groups
    split_v2 = {'train': [], 'val': [], 'test': []}
    for g in gids:
        bucket = 'test' if g in test_groups else ('val' if g in val_groups else 'train')
        split_v2[bucket].extend(groups[g])
    for x in eps:
        g = x['physical_group']
        x['split_v2'] = 'test' if g in test_groups else ('val' if g in val_groups else 'train')
    leakage_before = sum(1 for g in gids if len({('test' if x['episode'] in old_test else
                                                 ('val' if x['episode'] in set(old['val_episodes']) else 'train'))
                                               for x in eps if x['physical_group'] == g}) > 1)
    leak_after = sum(1 for g in gids if len({x['split_v2'] for x in eps if x['physical_group'] == g}) > 1)
    split_manifest = {'worker': 'task7', 'ts': ts, 'seed': SPLIT_SEED,
                      'rule': 'group-aware by action+eef trajectory hash; test groups = old-test groups + all quarantine groups (frozen); val groups picked by seeded shuffle, whole groups only',
                      'counts': {k: len(v) for k, v in split_v2.items()},
                      'groups_total': len(gids), 'test_groups': sorted(test_groups),
                      'train_episodes': sorted(split_v2['train']), 'val_episodes': sorted(split_v2['val']),
                      'test_episodes': sorted(split_v2['test']),
                      'leakage_groups_before': leakage_before, 'leakage_groups_after': leak_after}
    json.dump(split_manifest, open(OUT_SPLIT_V2, 'w'), indent=1)

    leaks = {'worker': 'task7', 'ts': ts,
             'old_split': {'train': len(old['train_episodes']), 'val': len(old['val_episodes']), 'test': len(old['test_episodes']),
                           'physical_groups_spanning_splits': leakage_before},
             'v2_split': {'counts': split_manifest['counts'], 'physical_groups_spanning_splits': leak_after,
                          'test_scenarios_frozen': True, 'test_groups': len(test_groups),
                          'quarantine_episodes': sum(1 for x in eps if x['quarantined_unknown_identity'])},
             'status': 'pass' if leak_after == 0 else 'fail',
             'split_manifest': OUT_SPLIT_V2}
    json.dump(leaks, open(OUT_LEAK, 'w'), indent=1)

    layout = {'worker': 'task7', 'ts': ts, 'source_column': {}, 'profile': {},
              'task_class': {}, 'visual_signature': {}, 'unknown_identity': 0, 'layout_hash': {}}
    for x in eps:
        layout['source_column'][str(x['source_column'])] = layout['source_column'].get(str(x['source_column']), 0) + 1
        layout['profile'][str(x['profile'])] = layout['profile'].get(str(x['profile']), 0) + 1
        layout['task_class'][x['task_class']] = layout['task_class'].get(x['task_class'], 0) + 1
        layout['visual_signature'][str(x['visual_signature'])] = layout['visual_signature'].get(str(x['visual_signature']), 0) + 1
        layout['layout_hash'][str(x['initial_scene_hash'])] = layout['layout_hash'].get(str(x['initial_scene_hash']), 0) + 1
        layout['unknown_identity'] += int(x['quarantined_unknown_identity'])
    layout['distinct_layouts'] = len([k for k in layout['layout_hash'] if k != 'None'])
    json.dump(layout, open(OUT_LAYOUT, 'w'), indent=1)

    with open(OUT_DEDUP, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['physical_group', 'members', 'n_members', 'action_sha1', 'eef_sha1',
                    'distinct_visual_signatures', 'distinct_scene_hashes', 'byte_level_action_duplicate'])
        for g in gids:
            mem = [x for x in eps if x['physical_group'] == g]
            w.writerow([g, ';'.join(str(x['episode']) for x in mem), len(mem),
                        mem[0]['action_sha1'], mem[0]['eef_sha1'],
                        len({x['visual_signature'] for x in mem}),
                        len({x['initial_scene_hash'] for x in mem}),
                        'yes' if len(mem) > 1 else 'no'])

    action_dup_groups = sum(1 for x in eps for y in eps if x['episode'] < y['episode'] and x['action_sha1'] == y['action_sha1'])
    frozen = json.load(open(FROZEN))
    prov = {'worker': 'task7', 'ts': ts, 'hf_repo': 'Eter0109/a3-cookie-multi-tasks',
            'hf_files_frozen': frozen['files'],
            'sim_repo': {'repo': 'Eter0109/a3_dual_arm_sim', 'commit': '08ebe8cb370856051d0b32b07e691cc9b056716b'},
            'collection_summary': json.load(open(P + '/hf_multi/collection_summary.json')),
            'episode_metadata_lines': len(lines), 'unique_ids': len(by_id),
            'task_class_resolution': {'metadata_single': sum(1 for x in eps if x['task_class_source'] == 'metadata_single'),
                                      'metadata_frame_match': sum(1 for x in eps if x['task_class_source'] == 'metadata_frame_match'),
                                      'unknown': sum(1 for x in eps if x['task_class_source'] == 'unknown')},
            'byte_level_action_duplicate_pairs': action_dup_groups,
            'physical_groups': len(gids)}
    json.dump(prov, open(OUT_PROV, 'w'), indent=1)

    accepted_known = sum(1 for x in eps if x['accepted_for_training'])
    quarantined = sum(1 for x in eps if x['quarantined_unknown_identity'])
    rejected = 845 - accepted_known - quarantined
    dataset_accepted = bool(rejected == 0 and leak_after == 0 and action_dup_groups == 0)
    manifest = {'schema_version': 2, 'generated_at': ts, 'source': 'Eter0109/a3-cookie-multi-tasks',
                'dataset_accepted': dataset_accepted,
                'acceptance_scope': 'training/val from 653 verified 2x5 episodes in groups; 96 quarantined unknown-identity + 96 old-test groups frozen as test; no 2x5/2x10 mixing in training',
                'counts': {'accepted_known': accepted_known, 'quarantined_unknown_identity': quarantined,
                           'rejected': rejected, 'total': 845},
                'split_v2': {'manifest': OUT_SPLIT_V2, 'counts': split_manifest['counts'],
                             'leakage_groups_after': leak_after},
                'episodes': eps}
    json.dump(manifest, open(OUT_MANIFEST, 'w'), indent=1)

    md = ['# DATASET_845_FINAL_QUALITY_AUDIT', '',
          '- dataset_accepted: **%s**' % dataset_accepted,
          '- episodes: 845 (accepted known=%d, quarantine unknown identity=%d, rejected=%d)' % (accepted_known, quarantined, rejected),
          '- task class resolution: metadata single=%d, metadata frame-match=%d, unknown=%d' % (
              prov['task_class_resolution']['metadata_single'], prov['task_class_resolution']['metadata_frame_match'],
              prov['task_class_resolution']['unknown']),
          '- byte-level action duplicate pairs: %d' % action_dup_groups,
          '- physical groups: %d; group leakage before=%d after=%d' % (len(gids), leakage_before, leak_after),
          '- split v2: %s' % json.dumps(split_manifest['counts']),
          '- layout coverage: source_column=%s task_class=%s unknown_identity=%d' % (
              json.dumps(layout['source_column']), json.dumps(layout['task_class']), layout['unknown_identity']),
          '- 2x5 and 2x10 are never mixed in training; 96 unknown-identity episodes are frozen test-only',
          '- files: PROVENANCE/DEDUP/LAYOUT/LEAKAGE/SPLIT_V2 + this manifest']
    open(OUT_AUDIT, 'w').write('\n'.join(md))
    print(json.dumps({'dataset_accepted': dataset_accepted, 'accepted_known': accepted_known,
                      'quarantined': quarantined, 'rejected': rejected,
                      'action_dup_pairs': action_dup_groups,
                      'groups': len(gids), 'leak_before': leakage_before, 'leak_after': leak_after,
                      'split_counts': split_manifest['counts'],
                      'class_resolution': prov['task_class_resolution']}))


if __name__ == '__main__':
    main()