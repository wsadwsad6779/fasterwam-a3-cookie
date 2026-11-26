#!/usr/bin/env python3
"""Bundle ALL eval videos on the pod into one named-by-experiment zip + manifest."""
import os, csv, zipfile, hashlib, json, time

ROOT = '/workspace/fwv1/eval_fingertip_v1'
OUTDIR = '/workspace/fwv1'
ZIP = os.path.join(OUTDIR, 'EVAL_VIDEOS_named_20261126.zip')
CSV = os.path.join(OUTDIR, 'EVAL_VIDEOS_MANIFEST_20261126.csv')

def exp_of(d):
    m = {
        'cmp_f0_3k': 'F0_milestone_3k', 'cmp_f0_6k': 'F0_milestone_6k', 'cmp_f0_9k': 'F0_milestone_9k',
        'cmp_f0_12k': 'F0_milestone_12k', 'cmp_f0_15k': 'F0_milestone_15k', 'cmp_f2_3k': 'F2_milestone_3k',
        'base_v31_thinpads': 'BASE_anchor_v31_thinpads',
        'step_003000': 'F2_15k_trial_step3000', 'step_006000': 'F2_milestone_6k', 'step_009000': 'F2_milestone_9k',
        'step_012000': 'F2_milestone_12k', 'step_015000': 'F2_milestone_15k',
        'diag15k_r10_grid': 'F2_15k_r10_grid', 'diag15k_r16_grid': 'F2_15k_r16_grid', 'diag15k_r32_grid': 'F2_15k_r32_grid',
        'h_eval_h32': 'F2_15k_H32', 'h_eval_h48': 'F2_15k_H48', 'h_eval_h64': 'F2_15k_H64',
        'f0_15k_r10': 'F0_15k_r10', 'f0_15k_r16': 'F0_15k_r16', 'f0_15k_r32': 'F0_15k_r32', 'f0_15k_relaxed': 'F0_15k_relaxed_r10',
    }
    if d in m: return m[d]
    if d.startswith('diag15k_r32_s'): return 'F2_15k_r32_' + d.split('_s')[-1]
    if d.startswith('diag15k_r16_s'): return 'F2_15k_r16_' + d.split('_s')[-1]
    if d.startswith('diag15k_r10_s'): return 'F2_15k_r10_' + d.split('_s')[-1]
    if d.startswith('diag15k_forcezero_s'): return 'F2_15k_forcezero_' + d.split('_s')[-1]
    if d.startswith('diag15k_skipfilm_s'): return 'F2_15k_skipfilm_' + d.split('_s')[-1]
    if d.startswith('aborted_'): return 'ABORTED_' + d
    if d.startswith('gbreq_'): return 'GREQ_' + d
    return d

rows = []
os.makedirs(OUTDIR, exist_ok=True)
with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
    for d in sorted(os.listdir(ROOT)):
        dp = os.path.join(ROOT, d)
        if not os.path.isdir(dp): continue
        exp = exp_of(d)
        for sd in sorted(os.listdir(dp)):
            sp = os.path.join(dp, sd)
            if not os.path.isdir(sp): continue
            for fn in sorted(os.listdir(sp)):
                if not fn.endswith('.mp4'): continue
                fp = os.path.join(sp, fn)
                seed = sd.replace('seed_', '')
                arc = '%s/seed_%s/%s' % (exp, seed, fn)
                z.write(fp, arc)
                rows.append({'experiment': exp, 'seed': seed, 'src_dir': d, 'arcname': arc, 'bytes': os.path.getsize(fp)})
    # manifest inside zip too
    mpath = CSV
    with open(mpath, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['experiment', 'seed', 'src_dir', 'arcname', 'bytes'])
        w.writeheader(); w.writerows(rows)
    z.write(mpath, '00_MANIFEST.csv')

sha = hashlib.sha256()
with open(ZIP, 'rb') as f:
    for c in iter(lambda: f.read(1 << 22), b''):
        sha.update(c)
print(json.dumps({'zip': ZIP, 'n_videos': len(rows), 'zip_bytes': os.path.getsize(ZIP), 'zip_sha256': sha.hexdigest(), 'manifest': CSV, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}))
