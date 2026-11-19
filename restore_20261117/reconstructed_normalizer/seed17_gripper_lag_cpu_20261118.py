import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
P = ROOT/'forensics_download'
s = np.load(P/'seed17_steps.npz', allow_pickle=False)
c = np.load(P/'seed17_original_chunks.npz', allow_pickle=False)
o = np.load(P/'cpu_seed17_state_force.npz', allow_pickle=False)
opening = o['observation.state'][:,7].astype(float)
command = s['handed_14d'][:,6]
delta = np.diff(opening)

def labels(d):
    out = np.zeros(len(d), np.int8); state = 0
    for t, v in enumerate(d):
        if v < -1e-4: state = 1
        elif v > 1e-4: state = -1
        out[t] = state
    return out

measured = labels(delta)
intent = np.zeros(600, np.int8)   # model pm1 sign: +1 close convention
for k in range(60):
    intent[10*k:10*k+10] = np.where(c[f'chunk_{k:03d}_pm1'][:10,6] >= 0, 1, -1)
executed = np.where(command >= 0, 1, -1).astype(np.int8)

dynamic = np.abs(delta) > 1e-4
lag_rows = []
for lag in range(-3, 4):
    if lag >= 0:
        a, b = intent[:600-lag], measured[lag:]
        m = dynamic[lag:]
    else:
        a, b = intent[-lag:], measured[:600+lag]
        m = dynamic[:600+lag]
    lag_rows.append({'lag_steps': lag, 'agree_all': int(np.sum(a == b)),
        'n_all': int(len(a)), 'agree_dynamic': int(np.sum((a == b) & m)), 'n_dynamic': int(m.sum()),
        'agree_rate_dynamic': float(np.sum((a == b) & m)/max(int(m.sum()),1))})

prev_intent = np.r_[intent[0], intent[:-1]]
hold_baseline = {'hold_previous_intent_vs_measured_agree': int(np.sum(prev_intent == measured)),
    'hold_rate': float(np.mean(prev_intent == measured))}

# Dynamics response: signed opening change per executed sign, and next-step reversal frequency.
resp = {'executed_close_steps': int(np.sum(executed == -1)), 'executed_open_steps': int(np.sum(executed == 1))}
for name, mask in [('close_steps', executed == -1), ('open_steps', executed == 1)]:
    resp[name+'_delta_mm_median'] = float(np.median(delta[mask])*1000)
    resp[name+'_delta_mm_p10_p90'] = (np.percentile(delta[mask], [10,90])*1000).tolist()
resp['measured_direction_follows_command_sign'] = float(np.mean((delta < 0) == (executed == -1)))
resp['command_reversals'] = int(np.sum(executed[1:] != executed[:-1]))
resp['measured_reversals'] = int(np.sum((delta[1:] > 1e-4) != (delta[:-1] > 1e-4)))

paths = [Path(__file__), P/'seed17_steps.npz', P/'seed17_original_chunks.npz', P/'cpu_seed17_state_force.npz']
report = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN',
    'scope': 'CPU arithmetic on stored seed17 episode; intent-versus-measured lag analysis. No model/simulator/network.',
    'sign_convention': 'pm1>=0 is intent +1 (close per label rule); Adapter inverts to command -1; measured opening decreases on command -1. pred_vs_executed exact negation is convention, not error.',
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'lag_analysis': lag_rows, 'hold_baseline': hold_baseline, 'dynamics': resp,
    'limits': ['Lag scan is descriptive on one episode and cannot distinguish policy error from plant lag/contact.', 'Deadband 1e-4 in normalized opening units.', 'No per-finger joint data; opening is a proxy.', 'One episode, one model, no counterfactual execution.']}
(ROOT/'SEED17_GRIPPER_LAG_CPU_20261118.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'lag_analysis': lag_rows, 'hold_baseline': hold_baseline, 'dynamics': resp}, indent=2))
