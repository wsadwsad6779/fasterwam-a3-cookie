import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import strict_production_pair as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'cpu_real_capture_regression'
OUT.mkdir(exist_ok=True)
source = S.source_paths()['original']
sys.path.insert(0, str(source.parent))
EC = S.load_file('cpu_evaluation_real', source.parent / 'evaluation_capture.py')
FIX = S.load_file('cpu_fixed_real', ROOT / 'execution_chain_capture_fixed.py')
EC.Capture = FIX.Capture
assets = {'weights': [str(Path(__file__))], 'config': [str(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json')], 'source': [str(source)]}
hashes = FIX.hash_files(assets)
class Env:
    def __init__(self):
        self.data = SimpleNamespace(**{k: np.zeros(2) for k in ('qpos','qvel','ctrl','act','qacc','sensordata')})
        self.state = np.zeros(16)
    def step(self, action):
        self.state[7] += action[6] * 0.01
        return {'observation.state': self.state.copy()}, 0, False, False, {'requested_action': action.copy(), 'applied_action': np.zeros(16)}
class Adapter:
    def convert(self, action):
        return np.r_[action, np.zeros(7)]
class Policy:
    def __init__(self):
        self.bridge = SimpleNamespace(denormalize_action=lambda a, p: a[None].copy())
        self.adapter = Adapter()
    def chunk(self, obs, frame, seed):
        raw = np.zeros((32,7))
        raw[:,6] = np.where(np.arange(32)%2, -0.7, 0.8)
        denorm = self.bridge.denormalize_action(raw, None)[0]
        denorm[:,6] = np.where(denorm[:,6]>=0,1,-1)
        return [self.adapter.convert(a) for a in denorm], None, {}
def episode(env, policy, seed, reset, **kw):
    obs = reset(env, seed)
    actions = policy.chunk(obs, frame=0, seed=seed)[0]
    for action in actions[:10]:
        env.step(action)
    return 'complete'
state = {'tag':'A'}
torch = SimpleNamespace(cuda=SimpleNamespace(get_rng_state_all=lambda: []))
runner = S.make_witnessed_runner(episode, EC, torch, OUT, state, assets, hashes)
for tag in ('A','B'):
    state['tag'] = tag
    assert runner(Env(), Policy(), 12, lambda e,s: {'observation.state': e.state.copy()}) == 'complete'
a = np.load(OUT / 'execution_A.npz', allow_pickle=False)
b = np.load(OUT / 'execution_B.npz', allow_pickle=False)
checks = {k: np.array_equal(a[k],b[k]) for k in a.files if k!='metadata'}
ma = json.loads(str(a['metadata']))
mb = json.loads(str(b['metadata']))
checks['artifact_identities_equal'] = ma['artifacts'] == mb['artifacts'] == hashes
checks['ten_executed_not_32'] = len(ma['steps']) == 10 and ma['chunks'][0]['predicted_rows'] == 32
checks['torch_not_imported'] = 'torch' not in sys.modules
report = {**S.D.MARKERS,'pass':all(checks.values()),'checks':checks,'scope':'SYNTHETIC_ENV_REAL_CAPTURE_REGRESSION_ONLY; fixtures not production weights'}
S.write_report(ROOT / 'CPU_REAL_CAPTURE_REGRESSION.json',report)
print(json.dumps({'pass':report['pass'],'checks_count':len(checks)}))
raise SystemExit(0 if report['pass'] else 1)
