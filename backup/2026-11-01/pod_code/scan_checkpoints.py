'''Phase 1 step A3: reconcile the inventory against the whole pod.

The first pass recorded 963 files. A plain `find /workspace -type f` returns 4933
files / 98.88 GB, so 3970 files (47.47 GB) had never been inventoried -- among them
the r3 full trainer state, every intermediate light checkpoint, both dagger
corpora and the upstream base models. This pass appends whatever is still missing
and labels each new file with a group, so the manifest describes the pod rather
than only the directories we happened to think about.

Excluded on purpose: podrun scratch (_cmd_*.sh, *.out), .cache, __pycache__, *.pyc
and .ipynb_checkpoints -- none of them is a research artifact.
'''
import json
import time
from pathlib import Path

W = Path('/workspace')
OUT = W / '_r6_stage/out'
LIST = OUT / 'backup_filelist.json'


def stamp(m):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(m))


def group_of(rel):
    r = rel
    if r.startswith('a3_train_runs/') and '/checkpoints/weights/' in r:
        return 'trainer_state_full'
    if r.startswith('a3_train_runs/') and '/checkpoints/light/' in r:
        run = r.split('/')[1]
        if run.startswith('chk') or 'crashed' in run:
            return 'light_ckpts_probe'
        return 'light_ckpts_series'
    if r.startswith('a3_data/a3_fw_dagger'):
        return 'dagger_corpus'
    if r.startswith('fasterwam_a3_bridge/bridge/outputs/dagger'):
        return 'dagger_raw_frames'
    if r.startswith('fasterwam_a3_bridge/bridge/outputs/'):
        return 'bridge_eval_outputs'
    if r.startswith('fasterwam_a3_bridge/'):
        return 'code_bridge'
    if r.startswith('FasterWAM/checkpoints/Wan-AI/'):
        return 'upstream_base_wan'
    if r.startswith('FasterWAM/checkpoints/fasterwam_release/'):
        return 'upstream_base_fasterwam'
    if r.startswith('FasterWAM/'):
        return 'code_upstream_fasterwam'
    if r.startswith('a3_dual_arm_sim/'):
        return 'code_sim'
    if r.startswith('fasterwam-a3-cookie/'):
        return 'code_cookie_repo'
    if r.endswith('.tar.gz') or r.endswith('.tgz'):
        return 'pod_archives'
    if '/' not in r and (r.endswith('.log') or r.endswith('.out')
                         or r.endswith('.csv') or r.endswith('.json')):
        return 'run_logs_root'
    if r.startswith(('_r4_stage/', '_r5_stage/', '_r6_stage/')):
        return 'stage_artifacts'
    return 'other'


def is_scratch(rel):
    seg = '/' + rel
    return ('/.cache/' in seg or '/__pycache__/' in seg
            or '/.ipynb_checkpoints/' in seg or rel.endswith('.pyc')
            or '_cmd_' in rel or rel.endswith('.out'))


def main():
    d = json.loads(LIST.read_text())
    ents = {e['path']: e for e in d['entries']}
    kept = list(d['entries'])
    before = len(kept)
    added = []
    skipped = []

    for p in sorted(W.rglob('*')):
        if not p.is_file():
            continue
        sp = str(p)
        if sp.startswith(str(OUT)) or '/_r6_stage/out/' in sp:
            continue
        rel = sp[len(str(W)) + 1:]
        if is_scratch(rel):
            skipped.append(rel)
            continue
        if sp in ents:
            continue
        st = p.stat()
        e = {'path': sp, 'bytes': st.st_size, 'mtime_utc': stamp(st.st_mtime),
             'suffix': p.suffix, 'groups': [group_of(rel)], 'referenced_by': False}
        ents[sp] = e
        kept.append(e)
        added.append(e)

    d['entries'] = kept
    d['widened_utc2'] = stamp(time.time())
    d['reconciled_from'] = 'full_listing.tsv sha256 cc587b68'
    d['added_second_pass'] = len(added)
    d['skipped_scratch_second_pass'] = len(skipped)
    LIST.write_text(json.dumps(d, indent=1))

    print('entries_before %d added %d now %d skipped_scratch %d' % (
        before, len(added), len(kept), len(skipped)))
    g = {}
    for e in added:
        k = e['groups'][0]
        a = g.setdefault(k, [0, 0])
        a[0] += 1
        a[1] += e['bytes']
    for k in sorted(g, key=lambda k: -g[k][1]):
        print('%-26s files=%-6d bytes=%d' % (k, g[k][0], g[k][1]))
    print('total_bytes_now %d' % sum(e['bytes'] for e in kept))


main()
