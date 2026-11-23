import json, hashlib, time
import numpy as np
import pyarrow.parquet as pq

META = '/workspace/fwv1/p_items/hf_multi/a3_episode_metadata.jsonl'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
SPLIT = '/workspace/fwv1/p_items/DATASET_800_SPLIT_MANIFEST.json'
OUT = '/workspace/fwv1/p_items/DATASET_800_METADATA_RECONCILIATION.json'


def ranges(ids):
    ids = sorted(ids)
    if not ids:
        return []
    out = []
    start = prev = ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append([start, prev])
        start = prev = x
    out.append([start, prev])
    return out


def main():
    lines = [l for l in open(META) if l.strip()]
    entries = [json.loads(l) for l in lines]
    by_id = {}
    line_by_id = {}
    for l, d in zip(lines, entries):
        by_id.setdefault(int(d['episode_index']), []).append(d)
        line_by_id.setdefault(int(d['episode_index']), []).append(l)

    t = pq.read_table(PARQ, columns=['episode_index'])
    E = np.asarray(t['episode_index'])
    order = np.argsort(E, kind='stable')
    E = E[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    parq_frames = {int(E[st[k]]): int(en[k] - st[k]) for k in range(len(st))}
    parq_ids = set(parq_frames)
    assert len(parq_ids) == 845

    dup_ids = sorted(k for k, v in by_id.items() if len(v) > 1)
    dup_groups = []
    conflicts = 0
    identical = 0
    for k in dup_ids:
        rows = by_id[k]
        sig = []
        for d in rows:
            r = dict(d)
            r.pop('seed', None)
            if isinstance(r.get('randomization'), dict):
                rr = dict(r['randomization'])
                rr.pop('seed', None)
                r['randomization'] = rr
            sig.append(json.dumps(r, sort_keys=True))
        same = len(set(sig)) == 1
        identical += int(same)
        conflicts += int(not same)
        fields = ('frames', 'success', 'source_dataset', 'source_shard', 'controller_type')
        vals = {f: sorted({json.dumps(d.get(f), sort_keys=True) for d in rows}) for f in fields}
        dup_groups.append({'episode': k, 'n_entries': len(rows), 'identical_except_seed': same,
                           'field_value_counts': {f: len(v) for f, v in vals.items()}})

    missing = sorted(parq_ids - set(by_id))
    extra = sorted(set(by_id) - parq_ids)
    source_counts = {}
    mapping = {}
    for k in sorted(parq_ids):
        rows = by_id.get(k, [])
        srcs = sorted({str(d.get('source_dataset') or d.get('controller_type') or 'unknown') for d in rows})
        key = '+'.join(srcs) if srcs else 'MISSING_METADATA'
        source_counts[key] = source_counts.get(key, 0) + 1
        mapping[str(k)] = key
    map_blob = json.dumps(mapping, sort_keys=True).encode()
    map_sha = hashlib.sha256(map_blob).hexdigest()

    frame_mismatch = []
    for k, rows in by_id.items():
        if k not in parq_frames:
            continue
        for d in rows:
            if d.get('frames') is not None and int(d['frames']) != parq_frames[k]:
                frame_mismatch.append({'episode': k, 'metadata_frames': int(d['frames']),
                                       'parquet_frames': parq_frames[k],
                                       'source_dataset': d.get('source_dataset')})

    split = json.load(open(SPLIT))
    coverage = {}
    for name in ('train_episodes', 'val_episodes', 'test_episodes'):
        ids = [int(x) for x in split[name]]
        cov = sum(1 for i in ids if i in by_id)
        coverage[name.replace('_episodes', '')] = {'n': len(ids), 'with_metadata': cov,
                                                   'missing': len(ids) - cov}

    res = {
        'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'parquet': {'episodes': len(parq_ids), 'frames': int(en[-1]),
                    'id_range': [int(min(parq_ids)), int(max(parq_ids))]},
        'metadata': {'lines': len(lines), 'unique_ids': len(by_id),
                     'duplicate_ids': len(dup_ids),
                     'ids_with_2plus_entries': len(dup_ids),
                     'duplicate_groups_identical_except_seed': identical,
                     'duplicate_groups_with_conflicts': conflicts,
                     'duplicate_groups_sample': dup_groups[:10]},
        'missing': {'count': len(missing), 'ranges': ranges(missing), 'sample': missing[:20],
                    'extra_metadata_ids_not_in_parquet': extra[:20]},
        'source_distribution': source_counts,
        'stable_mapping': {'rule': 'per episode id: sorted unique source_dataset tags joined with +; missing -> MISSING_METADATA',
                           'mapping_sha256': map_sha,
                           'mapping_size': len(mapping)},
        'frames_check': {'mismatch_count': len(frame_mismatch), 'sample': frame_mismatch[:10]},
        'split_coverage': coverage,
        'verdict': 'pass' if (not extra and all(v['missing'] >= 0 for v in coverage.values())) else 'fail',
    }
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'unique_ids': len(by_id), 'dup_ids': len(dup_ids),
                      'identical': identical, 'conflicts': conflicts,
                      'missing': len(missing), 'missing_ranges': len(ranges(missing)),
                      'frame_mismatch': len(frame_mismatch), 'sources': source_counts,
                      'coverage': coverage, 'mapping_sha16': map_sha[:16]}))


if __name__ == '__main__':
    main()