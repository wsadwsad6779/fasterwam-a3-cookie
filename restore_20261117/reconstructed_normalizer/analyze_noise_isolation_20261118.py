import hashlib
import json
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parent / 'forensics_download'
M = {'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL', 'original_stats_sha256': 'UNKNOWN'}
manifest = json.loads((P/'mechanismv3_download_manifest.json').read_text())
for x in manifest:
    assert hashlib.sha256((P/x['name']).read_bytes()).hexdigest() == x['sha256']
rows = []
for p in sorted(P.glob('mechanismv3_*.npz')):
    z = np.load(p, allow_pickle=False)
    j = json.loads(p.with_suffix('.json').read_text())
    assert hashlib.sha256(p.read_bytes()).hexdigest() == j['npz_sha256']
    seed = j['seed']
    baseline = np.load(P/f'mechanismv3_isolate_s{seed}_baseline.npz')
    bj = json.loads((P/f'mechanismv3_isolate_s{seed}_baseline.json').read_text())
    changed = [a['stream'] for a, b in zip(j['noise_draws'], bj['noise_draws']) if a['sha256'] != b['sha256']]
    expected = [s for s in ['video', 'action', 'force'] if s+'_noise' in j]
    assert changed == expected, (p.name, changed, expected)
    assert j['observation_sha256'] == bj['observation_sha256']
    assert np.array_equal(z['normalized_proprio'], baseline['normalized_proprio'])
    assert np.array_equal(z['force_history'], baseline['force_history'])
    assert np.array_equal(z['force_history_mask'], baseline['force_history_mask'])
    g = z['raw32'][:, 6]
    init = z['trace_before'][0, 0, :, 6]
    rows.append({'name': j['name'], 'changed_streams_verified': changed,
                 'flips10': int(np.sum(np.diff(g[:10]>=0))), 'flips32': int(np.sum(np.diff(g>=0))),
                 'gripper_max_abs_diff_baseline': float(np.abs(g-baseline['raw32'][:, 6]).max()),
                 'motion6_rms_diff_baseline': float(np.sqrt(np.mean((z['raw32'][:, :6]-baseline['raw32'][:, :6])**2))),
                 'future_force_rms_diff_baseline': float(np.sqrt(np.mean((z['force_future_newton']-baseline['force_future_newton'])**2))),
                 'initial_final_sign_agreement32': float(np.mean((init>=0)==(g>=0))),
                 'trace_flips10': j['trace_flip_counts_first10'], 'raw_gripper32': g.tolist()})
checks = []
for seed in [12, 17]:
    for label, oldlabel in [('baseline', 'A0'), ('all202', 'N202')]:
        x = np.load(P/f'mechanismv3_isolate_s{seed}_{label}.npz')
        y = np.load(P/f'gpu20261118_seed{seed}_frame0_{oldlabel}.npz')
        checks.append({'seed': seed, 'label': label, 'raw_exact': bool(np.array_equal(x['raw32'], y['raw32'])),
                       'force_exact': bool(np.array_equal(x['force_future_newton'], y['force_future_newton']))})
assert all(x['raw_exact'] and x['force_exact'] for x in checks)
a = np.load(P/'mechanismv3_isolate_s12_baseline.npz')
b = np.load(P/'mechanismv3_isolate_s17_baseline.npz')
initial_identity = {k: bool(np.array_equal(a[k], b[k])) for k in a.files if k.startswith('observation.') or k in ['force_history', 'force_history_mask', 'normalized_proprio']}
assert all(initial_identity.values())
out = {**M, 'hash_verified_files': len(manifest), 'records': rows, 'prior_result_exact_checks': checks,
       'seed12_seed17_initial_inputs_exact': initial_identity,
       'independent_observation_count': 1,
       'scope': 'Offline prediction only; changing noise is not an evaluated control repair',
       'inference': 'Cross-stream same-seed correlation is not necessary for fragmented modes in these two protocol-noise groups sharing one exactly identical initial observation. Action-noise changes alone select different early sign modes, but do not establish training-label causality.'}
(P/'NOISE_ISOLATION_ANALYSIS_20261118.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
print(json.dumps({'files': len(manifest), 'checks': checks, 'rows': [{k: r[k] for k in ['name', 'flips10', 'flips32', 'changed_streams_verified']} for r in rows]}, indent=2))
