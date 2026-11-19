import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import strict_production_pair as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cpu_launcher_regression'
OUT.mkdir(exist_ok=True)
checks = {}
checks['import_does_not_load_torch'] = 'torch' not in sys.modules
with patch.object(S, 'preflight', return_value={'pass': False, 'errors': ['synthetic failure']}), patch.object(S, 'load_runtime') as runtime:
    checks['failed_preflight_blocks_runtime'] = S.main(['--run-pair', '--out', str(OUT)]) == 1 and not runtime.called
with patch.object(S, 'preflight', return_value={'pass': True}), patch.object(S, 'load_runtime') as runtime:
    checks['preflight_only_blocks_runtime'] = S.main(['--preflight-only', '--out', str(OUT)]) == 0 and not runtime.called
checks['default_blocks_runtime'] = False
with patch.object(S, 'preflight', return_value={'pass': True}), patch.object(S, 'load_runtime') as runtime:
    checks['default_blocks_runtime'] = S.main(['--out', str(OUT)]) == 0 and not runtime.called
original = S.load_file('cpu_original_test', S.source_paths()['original'])
assets = {'weights': [str(Path(S.__file__))], 'config': [str(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json')], 'source': [str(S.source_paths()['original'])]}
hashes = original.hash_files(assets)
checks['synthetic_fixture_hashes_use_real_function'] = bool(hashes['weights'][0]['sha256'])
class Env:
    def __init__(self):
        self.data = SimpleNamespace(**{k: np.zeros(2) for k in ('qpos', 'qvel', 'ctrl', 'act', 'qacc', 'sensordata')})
    def step(self, action):
        return {'state': np.asarray(action)}, 0, False, False, {}
env = Env()
torch = SimpleNamespace(cuda=SimpleNamespace(get_rng_state_all=lambda: []))
seen = []
class Capture:
    artifacts = hashes
    def save(self, path):
        np.savez_compressed(path, synthetic=np.zeros(1))
def run_captured(runner, **kw):
    seen.append(kw['artifacts'])
    return runner(kw['env'], kw['policy'], kw['seed'], kw['reset']), Capture()
def episode(env, policy, seed, reset):
    reset(env, seed)
    env.step(np.ones(2))
    return 'synthetic result'
state = {'tag': 'A'}
witness = S.make_witnessed_runner(episode, SimpleNamespace(run_captured=run_captured), torch, OUT, state, assets, hashes)
for tag in ('A', 'B'):
    state['tag'] = tag
    before = env.step
    assert witness(env, object(), 12, lambda e, s: {'state': np.zeros(2)}) == 'synthetic result'
    assert env.step == before
checks['A_B_same_artifacts_object'] = len(seen) == 2 and seen[0] is assets and seen[1] is assets
for name, error in [('initialization', ValueError('synthetic init')), ('execution', RuntimeError('synthetic action'))]:
    def fail(*a, **kw):
        raise error
    witness = S.make_witnessed_runner(episode, SimpleNamespace(run_captured=fail), torch, OUT, state, assets, hashes)
    before = env.step
    try:
        witness(env, object(), 12, lambda e, s: {})
        checks[name + '_error_preserved'] = False
    except type(error) as exc:
        checks[name + '_error_preserved'] = exc is error and env.step == before
report = {**S.D.MARKERS, 'pass': all(checks.values()), 'checks': checks, 'scope': 'SYNTHETIC_LAUNCHER_REGRESSION_ONLY_NOT_REAL_WEIGHT_ACCEPTANCE', 'torch_imported': 'torch' in sys.modules}
S.write_report(ROOT / 'CPU_LAUNCHER_REGRESSION.json', report)
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['pass'] else 1)
