import json, os, time, hashlib
import numpy as np
import pyarrow.parquet as pq

META = '/workspace/fwv1/p_items/hf_multi/a3_episode_metadata.jsonl'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
FROZEN = '/workspace/fwv1/p_items/FROZEN_DATA_MANIFEST.json'
OUT_SPLIT = '/workspace/fwv1/p_items/DATASET_800_SPLIT_MANIFEST.json'
OUT_CONTACT = '/workspace/fwv1/p_items/DATASET_800_CONTACT_COVERAGE.json'
OUT_MD = '/workspace/fwv1/p_items/DATASET_800_SOURCE_AUDIT_T7.md'
SPLIT_SEED = 20261123


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
    recs = []
    with open(META) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    by_id = {}
    for d in recs:
        by_id.setdefault(int(d['episode_index']), []).append(d)
    t_ids = pq.read_table(PARQ, columns=['episode_index'])
    ids = sorted(set(int(x) for x in np.asarray(t_ids['episode_index']).tolist()))
    n = len(ids)
    assert n == 845, n
    src = {}
    multi = 0
    for k, lst in by_id.items():
        ss = sorted({str(d.get('source_dataset') or d.get('controller_type') or 'unknown') for d in lst})
        if len(ss) > 1:
            multi += 1
        src[k] = '+'.join(ss)
    strata = {}
    for k in ids:
        strata.setdefault(src.get(k, 'unknown'), []).append(k)
    meta_stats = {'lines': len(recs), 'unique_episode_ids': len(by_id),
                  'episodes_with_multi_source_tag': multi,
                  'parquet_ids_missing_metadata': len(set(ids) - set(by_id))}
    rng = np.random.default_rng(SPLIT_SEED)
    test, val, train = [], [], []
    for key in sorted(strata):
        ids = np.array(sorted(strata[key]))
        ids = ids[rng.permutation(len(ids))]
        n_test = int(round(0.10 * len(ids)))
        n_val = int(round(0.10 * len(ids)))
        test.extend(ids[:n_test].tolist())
        val.extend(ids[n_test:n_test + n_val].tolist())
        train.extend(ids[n_test + n_val:].tolist())
    test = sorted(test)
    val = sorted(val)
    train = sorted(train)
    all_ids = set(range(n))
    leakage = {
        'disjoint': len(set(train) & set(val)) == 0 and len(set(train) & set(test)) == 0 and len(set(val) & set(test)) == 0,
        'complete': set(train) | set(val) | set(test) == all_ids,
        'counts': [len(train), len(val), len(test)],
    }
    split = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             'source': 'Eter0109/a3-cookie-multi-tasks',
             'seed': SPLIT_SEED, 'strata': {k: len(v) for k, v in strata.items()},
             'metadata_stats': meta_stats,
             'n_train': len(train), 'n_val': len(val), 'n_test': len(test),
             'train_episodes': train, 'val_episodes': val, 'test_episodes': test,
             'leakage_check': leakage,
             'policy': {'train': 'optimization', 'val': 'milestone comparison/anchor', 'test': 'final independent only after model selection'}}
    assert leakage['disjoint'] and leakage['complete']
    json.dump(split, open(OUT_SPLIT, 'w'), indent=1)

    t = pq.read_table(PARQ, columns=['episode_index', 'observation.force'])
    E = np.asarray(t['episode_index'])
    order = np.argsort(E, kind='stable')
    E = E[order]
    fc = t['observation.force'].combine_chunks()
    flat = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    flat = flat[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    per = []
    for k in range(len(st)):
        F = flat[st[k]:en[k]]
        contact = (np.abs(F[:, 12:14]) > 0.05).any(axis=1)
        rcontact = (np.abs(F[:, 14:16]) > 0.05).any(axis=1)
        first = int(np.argmax(contact)) if contact.any() else None
        per.append({'episode': int(E[st[k]]), 'frames': int(en[k] - st[k]),
                    'contact_frames': int(contact.sum()), 'contact_any': bool(contact.any()),
                    'first_contact_frame': first,
                    'L_pad_max': float(np.abs(F[:, 12:14]).max()),
                    'R_pad_max': float(np.abs(F[:, 14:16]).max()),
                    'R_contact_frames': int(rcontact.sum())})
    with_contact = sum(1 for x in per if x['contact_any'])
    contact = {'worker': 'task7', 'ts': split['ts'], 'frames': int(len(flat)),
               'episodes': len(per),
               'episodes_with_L_pad_contact': with_contact,
               'episode_contact_rate': round(with_contact / len(per), 5),
               'frames_with_contact_frac': round(sum(x['contact_frames'] for x in per) / len(flat), 5),
               'R_pad_contact_episodes': sum(1 for x in per if x['R_contact_frames'] > 0),
               'L_pad_max_global': max(x['L_pad_max'] for x in per),
               'R_pad_max_global': max(x['R_pad_max'] for x in per),
               'per_episode': per[:20],
               'note': 'per_episode list truncated to 20; full per-episode values recomputable from parquet'}
    json.dump(contact, open(OUT_CONTACT, 'w'), indent=1)

    frozen = json.load(open(FROZEN))
    hashes = {}
    for p in PARQ:
        rel = 'data/chunk-000/' + os.path.basename(p)
        got = sha256_file(p)
        want = frozen['files'].get(rel, {}).get('sha256')
        hashes[rel] = {'sha256': got, 'frozen_sha256': want, 'match': got == want}
    meta_sha = sha256_file(META)
    meta_want = frozen['files'].get('a3_episode_metadata.jsonl', {}).get('sha256')
    hashes['a3_episode_metadata.jsonl'] = {'sha256': meta_sha, 'frozen_sha256': meta_want,
                                           'match': meta_sha == meta_want}
    md = ['# DATASET_800_SOURCE_AUDIT_T7', '',
          'Source: Eter0109/a3-cookie-multi-tasks (HF, LeRobot v2 layout); 845 episodes / 459177 frames.', '',
          '## File hash verification vs FROZEN_DATA_MANIFEST.json', '']
    for k, v in hashes.items():
        md.append('- %s: match=%s sha256=%s' % (k, v['match'], v['sha256'][:16]))
    md += ['', '## Metadata coverage',
           '- metadata lines %d, unique ids %d, multi-source ids %d, parquet ids missing metadata %d' % (
               meta_stats['lines'], meta_stats['unique_episode_ids'],
               meta_stats['episodes_with_multi_source_tag'], meta_stats['parquet_ids_missing_metadata']), '',
           '## Split (seed %d)' % SPLIT_SEED,
           '- train %d / val %d / test %d' % (len(train), len(val), len(test)),
           '- leakage check: %s' % json.dumps(leakage),
           '- strata: %s' % json.dumps(split['strata']), '',
           '## Contact coverage',
           '- episodes with L pad contact: %d/%d (%.1f%%)' % (with_contact, len(per), 100 * with_contact / len(per)),
           '- frames with contact: %.3f' % contact['frames_with_contact_frac'],
           '- R pad contact episodes: %d (expected 0 in dataset)' % contact['R_pad_contact_episodes'],
           '- L pad max |force|: %.2f N; R pad max: %.2f N' % (contact['L_pad_max_global'], contact['R_pad_max_global']), '',
           '## Existing provenance/dedup/diversity artifacts',
           '- DATASET_800_SOURCE_LOCATION.md, DATASET_800_PROVENANCE.json, DATASET_800_DIVERSITY_REPORT.md, DATASET_800_DEDUPLICATION.csv, DATASET_800_FORCE_COVERAGE.json',
           '- S2 re-verification: file hashes above; split isolation machine-checked']
    open(OUT_MD, 'w').write('\n'.join(md))
    print(json.dumps({'split': leakage, 'contact_rate': contact['episode_contact_rate'],
                      'R_contact_episodes': contact['R_pad_contact_episodes'],
                      'hashes_match': all(v['match'] for v in hashes.values())}))


if __name__ == '__main__':
    main()