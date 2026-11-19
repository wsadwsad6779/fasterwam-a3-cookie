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
command = s['handed_14d'][:,6]
delta = np.diff(opening)
executed = np.where(command >= 0, 1, -1)
pad_now = np.maximum(force[:-1,3], force[1:,3])
pad_out = np.maximum(force[:-1,4], force[1:,4])
contact = pad_now > 0.5

def bucket(step_mask):
    n = int(step_mask.sum())
    return {'n': n, 'pad_contact_n': int((step_mask & contact).sum()),
            'pad_contact_rate': float((step_mask & contact).sum()/max(n,1))}

close = executed == -1
open_ = executed == 1
up = delta > 1e-3
down = delta < -1e-3
static = ~(up | down)

buckets = {
 'close_and_opening_decreases': bucket(close & down),
 'close_and_opening_increases': bucket(close & up),
 'close_and_static': bucket(close & static),
 'open_and_opening_increases': bucket(open_ & up),
 'open_and_opening_decreases': bucket(open_ & down),
 'open_and_static': bucket(open_ & static)}

# Proper direction reversal: both consecutive deltas exceed 1mm with opposite signs.
rev = ((delta[1:] > 1e-3) & (delta[:-1] < -1e-3)) | ((delta[1:] < -1e-3) & (delta[:-1] > 1e-3))
rev_idx = np.flatnonzero(rev)
rev_contact = contact[1:][rev]

# Latch analysis: opening near limits while commanded further.
open_latched = (opening[:-1] > 1.05) & open_
closed_latched = (opening[:-1] < 0.0) & close

rows = []
for i in range(len(delta)):
    if up[i] or down[i]:
        rows.append({'i': i, 'frame_after': i+1, 'command': float(command[i]),
            'opening_before': float(opening[i]), 'opening_after': float(opening[i+1]),
            'delta_mm': float(delta[i]*1000), 'pad_inner_max_N': float(pad_now[i]),
            'pad_outer_max_N': float(pad_out[i]),
            'eef_z_m': float(o['observation.eef_pose'][i+1,2])})
contradict = [r for r in rows if (r['command'] < 0 and r['delta_mm'] > 1.0) or (r['command'] >= 0 and r['delta_mm'] < -1.0)]

at_open_limit = opening[:-1] > 1.0
at_closed_limit = opening[:-1] < 0.02
mid_range = ~(at_open_limit | at_closed_limit)

regime = {}
for name, mask in [('open_limit', at_open_limit), ('closed_limit', at_closed_limit), ('mid_range', mid_range)]:
    agree = (close & down) | (open_ & up)
    regime[name] = {'n': int(mask.sum()),
        'direction_agrees': int((agree & mask).sum()),
        'agreement_rate': float((agree & mask).sum()/max(int(mask.sum()),1)),
        'pad_contact_rate': float((contact & mask).sum()/max(int(mask.sum()),1)),
        'contradiction_n': int((mask & ~static & ~agree).sum())}

# Mid-range only: how often does the model intent match the realized direction?
mid_dynamic = mid_range & ~static
mid_agree = (close & down) | (open_ & up)

paths = [Path(__file__), P/'cpu_seed17_state_force.npz', P/'seed17_steps.npz', ROOT.parent/'step_040000/force_calib_v1.json']
report = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN',
 'scope': 'CPU arithmetic on stored seed17 episode; command sign versus measured opening direction and pad contact. No model/simulator/network/control.',
 'sign_convention': 'command -1 corresponds to the training label +1 close (Adapter inversion). Measured opening decreases on command -1 in the majority of steps.',
 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
 'thresholds': {'delta_1mm': 1e-3, 'contact_pad_N': 0.5},
 'buckets': buckets,
 'reversals': {'direction_reversal_count': int(rev.sum()), 'with_pad_contact': int(rev_contact.sum()),
    'first_10_indices': rev_idx[:10].tolist()},
 'latched_at_limits': {'open_above_1p05_while_open_command': int(open_latched.sum()),
    'closed_below_0_while_close_command': int(closed_latched.sum())},
 'regime_breakdown': regime,
 'mid_range_dynamic': {'n': int(mid_dynamic.sum()),
    'agreement_rate': float((mid_agree & mid_dynamic).sum()/max(int(mid_dynamic.sum()),1)),
    'contradiction_n': int((mid_dynamic & ~mid_agree).sum()),
    'contradiction_with_pad_contact_n': int((mid_dynamic & ~mid_agree & contact).sum())},
 'contradictions_count': len(contradict),
 'contradictions': contradict[:40],
 'limits': ['Contact classification uses pad channel >0.5N only; no contact-pair identity.', 'Opening is a normalized proxy, not metres.', 'A contradicted direction can be plant compliance or object interaction, not necessarily policy error.', 'Single episode, no counterfactual execution.']}
(ROOT/'SEED17_GRIPPER_CONTACT_COUPLING_CPU_20261118.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({k: report[k] for k in ['buckets','reversals','latched_at_limits','contradictions_count']}, indent=2))
print(json.dumps(contradict[:12], indent=1)[:1200])
