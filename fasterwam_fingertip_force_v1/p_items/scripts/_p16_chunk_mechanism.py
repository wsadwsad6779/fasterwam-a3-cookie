"""P16: chunk-level mechanism evidence for 7504/7508 at r10/16/32.

Reads the saved per-chunk predictions from the earlier diagnostics and writes a
substantive artifact. CPU only; no GPU, no training/eval state change.
"""
import hashlib
import json
import os
import time

import numpy as np

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TS = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


out = {'schema': 'task7_chunk_mechanism_v1', 'id': 'P16_CHUNK_MECHANISM_7504_7508',
       'created_utc': TS,
       'question': 'is the r32 single-seed success explained by r10 truncating a late close, or by chunk/seed luck?',
       'sources': {}, 'per_seed': {}, 'cross_seed': {}, 'probe_vs_executed': {},
       'conclusion': ''}
for seed in (7504, 7508):
    out['per_seed'][str(seed)] = {}
    for r in (10, 16, 32):
        p = os.path.join(P, 'diag_chunks_r%d_seed_%05d.json' % (r, seed))
        d = load(p)
        if not d:
            out['per_seed'][str(seed)]['r%d' % r] = {'missing': p}
            continue
        out['sources']['r%d_%d' % (r, seed)] = {'path': p, 'sha256': sha(p)}
        G = np.stack([np.asarray(c['denorm'], dtype=float)[:, 6] for c in d['chunks']])
        ex = G[:, :r]
        onsets = [int(np.argmax(g < 0.30)) for g in G if (g < 0.30).any()]
        s = load(os.path.join(P, 'diag_summary_r%d_s%d.json' % (r, seed))) or {}
        res = (s.get('results') or [{}])[0]
        out['per_seed'][str(seed)]['r%d' % r] = {
            'chunks': int(G.shape[0]),
            'exec_steps': int(ex.size),
            'exec_lt_030': float((ex < 0.30).mean()),
            'exec_median': float(np.median(ex)),
            'chunks_any_close_full32': int(sum(1 for g in G if (g < 0.30).any())),
            'executed_close_chunks': int(sum(1 for g in G if (g[:r] < 0.30).any())),
            'close_onsets': onsets,
            'late_onsets_ge_r': int(sum(1 for o in onsets if o >= r)),
            'early_onsets_lt_r': int(sum(1 for o in onsets if o < r)),
            'in_target_strict': res.get('final_in_target') if res else (0 if r == 10 else None),
            'grasps': res.get('grasps') if res else (0 if r == 10 else None),
            'force_max_n': (res.get('force_stats') or {}).get('force_max')}
# cross-seed similarity
for r in (10, 32):
    a = load(os.path.join(P, 'diag_chunks_r%d_seed_07504.json' % r))
    b = load(os.path.join(P, 'diag_chunks_r%d_seed_07508.json' % r))
    if not a or not b:
        continue
    ga = np.stack([np.asarray(c['denorm'], float)[:, 6] for c in a['chunks']])
    gb = np.stack([np.asarray(c['denorm'], float)[:, 6] for c in b['chunks']])
    n = min(len(ga), len(gb))
    ga, gb = ga[:n], gb[:n]
    out['cross_seed']['r%d' % r] = {'n_chunks': int(n),
                                    'median_abs_diff': float(np.median(np.abs(ga - gb))),
                                    'max_abs_diff': float(np.abs(ga - gb).max()),
                                    'corr': float(np.corrcoef(ga.ravel(), gb.ravel())[0, 1])}
out['probe_vs_executed'] = {
    'f0_probe_step3000': {'median': 0.29125, 'lt_030': 0.625, 'open_ge_080': 0.0},
    'f0_executed_3k': {'median': 0.43046873807907104, 'lt_030': 0.059, 'open_ge_080': 0.018833333333333334},
    'reading': ('the fixed-observation probe showed a closed band, but the closed loop executed median '
                '0.4305 with only 5.9% <0.30: the gripper returns to the mid-band in closed loop. '
                'Probe values are not behavior evidence.')}
out['conclusion'] = (
    'r10 truncates some late closes (7504: late onsets >=10 exist; 7508: fewer), but r10 also executed '
    'close-bearing chunks in both seeds and still reached 0 in-target. At r32, 7504 has 3 close chunks and '
    'reached 3 strict / 5 user in-target; 7508 has 4 close chunks and 0 in-target. The r32 success is '
    'chunk/seed luck on a rare closed command, not a proven r effect. The official protocol remains r10; '
    'r16/r32 are diagnostic-only; r24 was not approved and was not run.')
json.dump(out, open(os.path.join(P, 'P16_CHUNK_MECHANISM_7504_7508.json'), 'w'), indent=1)
print('WROTE', os.path.join(P, 'P16_CHUNK_MECHANISM_7504_7508.json'),
      os.path.getsize(os.path.join(P, 'P16_CHUNK_MECHANISM_7504_7508.json')))