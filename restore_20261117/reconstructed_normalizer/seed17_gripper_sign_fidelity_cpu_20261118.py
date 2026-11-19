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
assert len(opening) == len(command)+1

def labels_from(delta):
    out = np.zeros(len(delta), dtype=np.int8)
    state = 0
    for t, d in enumerate(delta):
        if d < -1e-4:
            state = 1
        elif d > 1e-4:
            state = -1
        out[t] = state
    return out

measured_label = labels_from(np.diff(opening))
command_sign = np.where(command >= 0, 1, -1).astype(np.int8)
replans = []
for k in range(len(command)//10):
    sl = slice(10*k, 10*k+10)
    pred = np.where(c[f'chunk_{k:03d}_pm1'][:10,6] >= 0, 1, -1).astype(np.int8)
    replans.append({'chunk':k,'predicted_sign10':pred.tolist(),
        'executed_sign10':command_sign[sl].tolist(),
        'measured_label10':measured_label[sl].tolist(),
        'pred_vs_executed_agree':int(np.sum(pred == command_sign[sl])),
        'pred_vs_measured_agree':int(np.sum(pred == measured_label[sl])),
        'executed_vs_measured_agree':int(np.sum(command_sign[sl] == measured_label[sl])),
        'measured_delta10_mm':(np.diff(opening)[sl]*1000).tolist()})
tot = lambda key: int(sum(r[key] for r in replans))
report = {'NORMALIZER_IDENTITY':'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL','original_stats_sha256':'UNKNOWN',
 'scope':'CPU arithmetic on stored seed17 episode; predicted chunks versus executed commands versus measured opening. No model/simulator/network.',
 'label_rule':'opening decrease>1e-4 -> +1 close; increase>1e-4 -> -1 open; unchanged -> inherit previous within episode.',
 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),P/'seed17_steps.npz',P/'seed17_original_chunks.npz',P/'cpu_seed17_state_force.npz']},
 'totals':{'replans':len(replans),'steps':len(command),
   'pred_vs_executed_agree_600':tot('pred_vs_executed_agree'),
   'pred_vs_measured_agree_600':tot('pred_vs_measured_agree'),
   'executed_vs_measured_agree_600':tot('executed_vs_measured_agree'),
   'executed_sign_flips':int(np.sum(command_sign[1:]!=command_sign[:-1])),
   'measured_label_flips':int(np.sum(measured_label[1:]!=measured_label[:-1])),
   'measured_open_to_close_events':int(np.sum((measured_label[1:]==1)&(measured_label[:-1]==-1))),
   'measured_close_to_open_events':int(np.sum((measured_label[1:]==-1)&(measured_label[:-1]==1))),
   'opening_range':[float(opening.min()),float(opening.max())],
   'opening_abs_delta_mm_percentiles':np.percentile(np.abs(np.diff(opening))*1000,[50,90,99,100]).tolist()},
 'replans':replans,
 'limits':['This compares stored predictions and measured opening only; it does not determine whether a flip released or retained a cookie.', 'Measured signs use a 1e-4 m-normalized deadband as in the recovered label builder; it is not a metric distance.', 'No fingertip joint positions are recorded, so opening is a normalized proxy.']}
(ROOT/'SEED17_GRIPPER_SIGN_FIDELITY_CPU_20261118.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps(report['totals'],indent=2))
print(json.dumps(replans[:3]+replans[6:8]+replans[7:8],indent=1)[:900])
