import glob
import json
import os
import sys
import time

import numpy as np

ROOT = '/workspace/fwv1/eval_fingertip_v1'
ARCH = '/root/fwv1_checkpoint_archive'
OUT = '/workspace/fwv1/p_items/F0_F2_COMPARISON_3K.json'
MD = '/workspace/fwv1/p_items/F0_F2_COMPARISON_3K.md'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def resolve(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def summarize(name, sub, ckpt_candidates):
    sp = os.path.join(ROOT, sub, 'SUMMARY_step_3000.json')
    s = json.load(open(sp))
    ckpt = resolve(ckpt_candidates)
    vals = []
    for p in glob.glob(os.path.join(ROOT, sub, 'seed_*', 'events.json')):
        d = json.load(open(p))
        for r in d['rows']:
            vals.append(r['act_grip_raw'])
    v = np.asarray(vals)
    res = s.get('results', [])
    return {
        'name': name, 'checkpoint_used': ckpt, 'summary': sp,
        'summary_checkpoint': s.get('checkpoint'),
        'checkpoint_sha256': s.get('checkpoint_sha256'),
        'exact_fill_count': s.get('exact_fill_count'), 'avg_in_target': s.get('avg_in_target'),
        'grasps_mean': float(np.mean([r.get('grasps', 0) for r in res])) if res else None,
        'grasps_total': int(sum(r.get('grasps', 0) for r in res)),
        'close_frac': (s.get('action_stats_mean') or {}).get('grip_close_frac'),
        'open_cmd_frac': (s.get('action_stats_mean') or {}).get('open_cmd_frac'),
        'reopen_mean': (s.get('action_stats_mean') or {}).get('reopen_events_mean'),
        'reopen_min': min([(r.get('action_stats') or {}).get('reopen_events', 0) for r in res]) if res else None,
        'carry_force_frac_gt_0p5': (s.get('force_stats_mean') or {}).get('carry_force_frac_gt_0p5'),
        'contact_frac': (s.get('force_stats_mean') or {}).get('contact_frac'),
        'force_max_mean': (s.get('force_stats_mean') or {}).get('force_max'),
        'eef_path_m': (s.get('action_stats_mean') or {}).get('eef_path_m'),
        'executed_grip_lt_0p30': float((v < 0.30).mean()) if len(v) else None,
        'executed_grip_median': float(np.median(v)) if len(v) else None,
        'executed_grip_lt_0p30_28band': float(((v >= 0.27) & (v <= 0.29)).mean()) if len(v) else None,
        'executed_steps': int(len(v)),
        'videos_dir': os.path.join(ROOT, sub)}


f0 = summarize('F0_step3000', 'cmp_f0_3k',
               [os.path.join(ARCH, 'f0_step_003000.pt'),
                '/workspace/fwv1/f0_v3_baseline/step_003000.pt'])
f2 = summarize('F2_step3000', 'cmp_f2_3k',
               [os.path.join(ARCH, 'step_003000.pt'),
                '/workspace/fwv1/train_runs/t7_21k_v3run/step_003000.pt'])
result = {'ts': TS,
          'protocol': 'same V3.1 scenes, seeds 7100-7109, 600 steps, eval script 37376d2c, frozen sim f237b766 / config 62f6c863 / freeze cef07926',
          'pairs': [f0, f2]}
if f2.get('summary_checkpoint') and not f2.get('checkpoint_sha256'):
    result['protocol_error'] = ('F2 summary has no checkpoint_sha256: the eval checkpoint path was missing and '
                                'build_model silently used the base model. This pair is NOT F2 3k.')
if f0.get('executed_grip_lt_0p30') is not None and f2.get('executed_grip_lt_0p30') is not None:
    c0, c2 = f0['executed_grip_lt_0p30'], f2['executed_grip_lt_0p30']
    if c0 >= 0.25 and c2 < 0.10:
        verdict = 'F0 closes substantially more often than F2: the force bundle is associated with the open/mid bias at 3k'
    elif c0 < 0.10 and c2 < 0.10:
        verdict = 'both F0 and F2 emit few closed-band commands at 3k: shared continuous supervision is the leading problem'
    elif c0 < 0.10 and c2 >= 0.25:
        verdict = 'F2 closes more often than F0 at 3k: opposite of a simple force-negative-transfer story'
    else:
        verdict = 'mixed closure rates: need more steps/scenes'
    result['verdict'] = verdict
json.dump(result, open(OUT, 'w'), indent=1)
lines = ['# F0 vs F2 step-3000 paired evaluation', '',
         '- ts: `%s`' % TS,
         '- protocol: %s' % result['protocol'],
         '- F0 checkpoint sha: `%s`' % f0.get('checkpoint_sha256'),
         '- F2 checkpoint sha: `%s`' % f2.get('checkpoint_sha256'), '']
if result.get('protocol_error'):
    lines += ['**PROTOCOL ERROR**: %s' % result['protocol_error'], '']
lines += ['| metric | F0 | F2 |', '|---|---|---|']
for k in ['exact_fill_count', 'avg_in_target', 'grasps_total', 'reopen_mean', 'reopen_min',
          'carry_force_frac_gt_0p5', 'contact_frac', 'force_max_mean',
          'executed_grip_lt_0p30', 'executed_grip_lt_0p30_28band', 'executed_grip_median',
          'close_frac', 'open_cmd_frac', 'eef_path_m']:
    lines.append('| %s | %s | %s |' % (k, f0.get(k), f2.get(k)))
lines += ['', 'verdict: %s' % result.get('verdict', 'n/a'), '',
          'videos: %s | %s' % (f0['videos_dir'], f2['videos_dir'])]
open(MD, 'w').write('\n'.join(lines) + '\n')
print(json.dumps(result))
if result.get('protocol_error'):
    sys.exit(2)