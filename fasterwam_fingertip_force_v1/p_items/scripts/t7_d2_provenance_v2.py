"""Enrich the accepted manifest with per-episode provenance pointers:
HF dataset revision sha, sim repo commit, and source component mapping."""
import json
import os
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import HfApi

P = '/workspace/fwv1/p_items'
MAN = P + '/DATASET_845_ACCEPTED_MANIFEST.json'
OUT = P + '/DATASET_845_PROVENANCE_V2.json'
SIM_COMMIT = '08ebe8cb370856051d0b32b07e691cc9b056716b'


def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    api = HfApi()
    rev = None
    try:
        info = api.dataset_info('Eter0109/a3-cookie-multi-tasks')
        rev = getattr(info, 'sha', None)
    except Exception as ex:
        rev = 'unavailable:' + type(ex).__name__
    summary = json.load(open(P + '/hf_multi/collection_summary.json'))
    comps = summary.get('components', [])
    comp_by_repo = {}
    for c in comps:
        rid = c.get('repo_id') or c.get('repo') or ''
        comp_by_repo[rid] = c
    man = json.load(open(MAN))
    counts = {}
    for e in man['episodes']:
        src = e.get('source_dataset') or ''
        comp = comp_by_repo.get(src, {})
        e['provenance'] = {
            'hf_repo': 'Eter0109/a3-cookie-multi-tasks',
            'hf_revision': rev,
            'sim_repo': 'Eter0109/a3_dual_arm_sim',
            'sim_commit': SIM_COMMIT,
            'source_dataset': src,
            'source_component_root': comp.get('root'),
            'source_component_target': comp.get('target') or comp.get('target_count'),
            'collection_seed': e.get('seed'),
        }
        counts[src or 'MISSING'] = counts.get(src or 'MISSING', 0) + 1
    man['provenance_enriched_at'] = ts
    json.dump(man, open(MAN, 'w'), indent=1)
    prov = {'worker': 'task7', 'ts': ts, 'hf_repo': 'Eter0109/a3-cookie-multi-tasks',
            'hf_revision': rev, 'sim_repo': 'Eter0109/a3_dual_arm_sim', 'sim_commit': SIM_COMMIT,
            'components': [{k: c.get(k) for k in ('repo_id', 'repo', 'root', 'target', 'target_count', 'task_instruction')} for c in comps],
            'episode_counts_by_source_dataset': counts,
            'per_episode_fields_added': ['hf_revision', 'sim_commit', 'source_component_root',
                                          'source_component_target', 'collection_seed'],
            'manifest': MAN}
    json.dump(prov, open(OUT, 'w'), indent=1)
    print(json.dumps({'hf_revision': rev, 'components': len(comps), 'counts': counts}))


if __name__ == '__main__':
    main()