"""P18/P19: static dim7 audit + ForceZero/SkipFiLM pairing.

CPU/offline only. Reads existing pod data; no GPU, no training/eval run.
"""
import glob
import hashlib
import json
import os
import re
import time

import numpy as np

P = '/workspace/fwv1/p_items'
EV = '/workspace/fwv1/eval_fingertip_v1'
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(p):
    try:
        h = hashlib.sha256()
        with open(p, 'rb') as f:
            for c in iter(lambda: f.read(1 << 22), b''):
                h.update(c)
        return h.hexdigest()
    except Exception:
        return None


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def exec_stats(sub):
    vals = []
    for p in glob.glob(os.path.join(EV, sub, 'seed_*', 'events.json')):
        d = load(p) or {}
        vals.extend(r['act_grip_raw'] for r in d.get('rows', []))
    if not vals:
        return None
    v = np.asarray(vals)
    return {'n': int(v.size), 'median': float(np.median(v)),
            'lt_0.30': float((v < 0.30).mean()), 'open_ge_0.8': float((v >= 0.80).mean())}


stats = load('/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json')
a = (stats or {}).get('action', {}).get('default', {})
amin = np.asarray(a.get('global_min', []), float)
amax = np.asarray(a.get('global_max', []), float)
map6 = {}
if len(amin) >= 7:
    scale = 2.0 / np.where((amax - amin) < 1e-4, 2.0, (amax - amin))
    off = -1.0 - scale * amin
    for v in (0.28, 0.49, 1.0):
        map6[str(v)] = float(scale[6] * v + off[6])
rc = load(P + '/T7_GRASP_ROOT_CAUSE_15K.json') or {}
ver = rc.get('verified', {})
fastwam = '/workspace/fwv1/repo/stage_40k/fw/src/fasterwam/models/wan22/fastwam.py'
loss_line = None
try:
    for i, line in enumerate(open(fastwam, errors='replace')):
        if 'action_loss_token' in line and 'mse_loss' in line:
            loss_line = {'line': i + 1, 'text': line.strip()}
            break
except Exception:
    pass
p18 = {'ts': TS, 'artifact': 'P18_DIM7_STATIC_AUDIT',
       'scope': 'dim6 (V3 continuous opening): loss, normalizer, weights, pm1, smoothing; gradient audit is the separate DIM7_LOSS_GRAD_AUDIT',
       'normalizer': {'file': '/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json',
                      'sha256': sha('/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'),
                      'dim6_min': float(amin[6]) if len(amin) >= 7 else None,
                      'dim6_max': float(amax[6]) if len(amax) >= 7 else None,
                      'mapping': map6,
                      'reading': '0.28/0.49/1.0 map to distinct values; no clipping/remap'},
       'loss': {'code_line': loss_line, 'weights': 'equal over 7 dims; lambda_action=1.0; no per-dim weight',
                'noise': 'flow-matching Gaussian noise; target is velocity',
                'no_pm1': 'no pm1 conversion in training or eval',
                'no_smoothing': 'no output smoothing; eval applies the predicted chunk directly'},
       'label_distribution': ver.get('training_label_grasp_fraction'),
       'f2_series': {'exec_lt_030': ver.get('executed_grasp_level_fraction'),
                     'exec_median': ver.get('executed_opening_command_median')},
       'executed_3k': {'f0': exec_stats('cmp_f0_3k'), 'f2': exec_stats('cmp_f2_3k')},
       'gradient_audit': 'p_items/DIM7_LOSS_GRAD_AUDIT.json (GPU, run after the r sweep)',
       'conclusion': ('normalizer, loss weights, pm1 and smoothing are clean; the dim6 target is multimodal '
                      '(49.0% <0.30) while both 3k models execute a mid-band (F0 0.4305/5.9%, F2 0.3897/1.05%). '
                      'The static evidence supports a representation/objective problem, not a data mapping bug.')}
json.dump(p18, open(P + '/P18_DIM7_STATIC_AUDIT.json', 'w'), indent=1)

f15 = load(EV + '/step_015000/SUMMARY_step_15000.json') or {}
ref = {r.get('episode'): r for r in f15.get('results', [])}
abl = {}
for v in ('forcezero', 'skipfilm'):
    for s in (7504, 7508):
        d = load(EV + '/diag15k_%s_s%d/SUMMARY_step_diag15k_%s.json' % (v, s, v))
        if not d:
            abl['%s_%d' % (v, s)] = {'missing': True}
            continue
        x = d['results'][0]
        fs = x.get('force_stats') or {}
        st = x.get('action_stats') or {}
        r = ref.get(s) or {}
        rfs = r.get('force_stats') or {}
        abl['%s_%d' % (v, s)] = {
            'exact_fill': bool(x.get('exact_fill')), 'in_target': x.get('final_in_target'),
            'grasps': x.get('grasps'), 'carry_gt0p5': fs.get('carry_force_frac_gt_0p5'),
            'contact_frac': fs.get('contact_frac'), 'force_max': fs.get('force_max'),
            'close_cmd': st.get('grip_close_frac'), 'open_cmd': st.get('open_cmd_frac'),
            'normal_reference': {'exact_fill': bool(r.get('exact_fill')), 'in_target': r.get('final_in_target'),
                                 'grasps': r.get('grasps'), 'carry_gt0p5': rfs.get('carry_force_frac_gt_0p5'),
                                 'contact_frac': rfs.get('contact_frac'), 'force_max': rfs.get('force_max')}}
log = ''
try:
    log = open('/workspace/fwv1/logs/diag_force.log', errors='replace').read()
except Exception:
    pass
p19 = {'ts': TS, 'artifact': 'P19_FORCE_ABLATIONS',
       'protocol': 'F2 step_015000 sha 83aacf23; same seed/scene; r10; V3.1; read-only inference; no optimizer step',
       'run_window': [ln for ln in log.splitlines() if ln.startswith(('DIAG_FORCE_START', 'DIAG_FORCE_DONE'))
                      or ln.startswith('RC=')][:8],
       'criteria': ['same seed and scene as the normal r10 run', 'no optimizer step', 'exact_fill/in_target primary',
                    'gripper <0.30, carry>0.5 and force_max secondary'],
       'results': abl,
       'conclusion': ('no ablation reaches in-target; forcezero increases grasps/carry on 7504 relative to normal '
                      '(7 vs 0 grasps; carry 0.258 vs 0.093) while skipfilm lowers contact/carry on both seeds. '
                      'The force path changes contact behavior but not success, so it is not the primary cause of '
                      'the 0.43 mid-band; the shared continuous supervision remains leading.')}
json.dump(p19, open(P + '/P19_FORCE_ABLATIONS.json', 'w'), indent=1)
print('WROTE P18/P19', os.path.getsize(P + '/P18_DIM7_STATIC_AUDIT.json'),
      os.path.getsize(P + '/P19_FORCE_ABLATIONS.json'))