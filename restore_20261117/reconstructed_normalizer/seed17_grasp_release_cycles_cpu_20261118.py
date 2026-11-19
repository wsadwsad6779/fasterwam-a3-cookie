import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT/'forensics_download'
o = np.load(P/'cpu_seed17_state_force.npz', allow_pickle=False)
s = np.load(P/'seed17_steps.npz', allow_pickle=False)
cal = json.loads((ROOT.parent/'step_040000/force_calib_v1.json').read_text())
force = o['observation.force'][:, cal['channels']].astype(float)
opening = o['observation.state'][:,7].astype(float)
pose = o['observation.eef_pose'][:,:3].astype(float)
command = s['handed_14d'][:,6]
n = len(command)

# Closed/open regimes with hysteresis: closed below 0.25, open above 0.65.
state = np.zeros(n+1, np.int8)  # 0 unknown, 1 closed, 2 open
cur = 0
for t in range(n+1):
    if opening[t] < 0.25:
        cur = 1
    elif opening[t] > 0.65:
        cur = 2
    state[t] = cur

segments = []
start = 0
for t in range(1, n+1):
    if state[t] != state[t-1]:
        segments.append({'state': int(state[t-1]), 'f_start': start, 'f_end': t-1, 'len': t-start})
        start = t
segments.append({'state': int(state[n]), 'f_start': start, 'f_end': n, 'len': n-start+1})

closed_runs = [g for g in segments if g['state'] == 1 and g['len'] >= 3]
open_runs = [g for g in segments if g['state'] == 2]
cycles = []
for run in closed_runs:
    f0, f1 = run['f_start'], run['f_end']
    z = pose[f0:f1+1, 2]
    pad = np.max(force[f0:f1+1][:,3:5], axis=1)
    # release = first transition closed->open after this run
    rel = next((g for g in segments if g['f_start'] == f1+1 and g['state'] == 2), None)
    if rel is None:
        continue
    r0 = rel['f_start']
    pre_pad = float(np.max(pad[-3:])) if len(pad) >= 3 else float(np.max(pad))
    rel_pad = float(np.max(force[r0:r0+3][:,3:5])) if r0 < n else 0.0
    cycles.append({'closed_frames': [f0, f1], 'closed_len': run['len'],
        'z_min_m': float(z.min()), 'z_max_m': float(z.max()), 'z_release_m': float(pose[r0,2]),
        'z_change_m': float(pose[r0,2]-z.min()),
        'pad_before_release_N': pre_pad, 'pad_at_release_N': rel_pad,
        'release_frame': int(r0), 'release_gap_steps': int(r0-f1-1)})

z_all = pose[:,2]
report = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN',
 'scope': 'CPU descriptive segmentation of stored seed17 episode; grasp-close runs and following releases with EEF height and pad force. No contact-pair identity, no simulator.',
 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), P/'cpu_seed17_state_force.npz', P/'seed17_steps.npz', ROOT.parent/'step_040000/force_calib_v1.json']},
 'definitions': {'closed': 'opening<0.25', 'open': 'opening>0.65', 'closed_run_min_len': 3,
   'pad_before_release': 'max pad channel of last3 closed frames', 'pad_at_release': 'max pad channel of release frame and next2'},
 'eef_z_percentiles_m': np.percentile(z_all, [0,10,50,90,100]).tolist(),
 'segment_counts': {'closed_runs_ge3': len(closed_runs), 'open_runs': len(open_runs)},
 'cycles': cycles,
 'limits': ['No direct cookie/box occupancy; releases are inferred from opening transitions only.', 'EEF site z is not cookie z; no object height or contact pair.', 'Segmentation thresholds are descriptive, not a task-success metric.', 'Pad force zero can occur without contact with the target object.']}
(ROOT/'SEED17_GRASP_RELEASE_CYCLES_CPU_20261118.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'eef_z_percentiles_m': report['eef_z_percentiles_m'], 'segment_counts': report['segment_counts'], 'cycle_count': len(cycles)}, indent=2))
for c in cycles:
    print(json.dumps(c))
