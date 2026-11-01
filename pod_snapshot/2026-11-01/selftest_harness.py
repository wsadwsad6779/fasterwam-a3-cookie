import sys, glob, types, importlib.util
import numpy as np

BR = '/workspace/fasterwam_a3_bridge/bridge'
NX = BR + '/_next'
sys.path.insert(0, NX)
sys.path.insert(0, BR)

fake = types.ModuleType('infer_a3_worker')
def _seed(s, get_worker_init_fn=True):
    np.random.seed(int(s))
fake.set_global_seed = _seed
sys.modules['infer_a3_worker'] = fake

import common

class Stub:
    name = 'lora'
    calls = 0
    gripper_mode = 'pm1'
    def chunk(self, obs, seed):
        self.calls += 1
        return np.zeros((16, 7))
    def adapted(self, chunk7, obs):
        a = np.zeros((len(chunk7), 14))
        a[:, 6] = -1.0
        a[:, 13] = -1.0
        return a.tolist()

def stub_load(name, ckpt=None, stats=None, instruction=None, mode=None):
    return Stub()

def stub_desc(name, ckpt=None, stats=None):
    return {'model': name,
            'checkpoint': {'path': str(ckpt), 'size': 1, 'sha256_head': 'stub'},
            'normalizer': {}}

common.load_policy = stub_load
common.describe_policy = stub_desc

def load_mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

new = load_mod(NX + '/run_matrix.py', 'rm_new')
old = load_mod(NX + '/run_matrix_orig_check.py', 'rm_old')
CKP = '/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt'

def run(mod, out):
    sys.argv = ['run_matrix.py', '--model', 'lora', '--controls', 'C0,C5',
                '--seeds', '11', '--max-steps', '20', '--out', out, '--ckpt', CKP]
    mod.main()

run(new, '/workspace/_selftest/new')
run(old, '/workspace/_selftest/old')

print('---- compare ----')
for f in sorted(glob.glob('/workspace/_selftest/new/lora_*_*.npz')):
    seed = int(f[:-4].split('_')[-1])
    ctrl = f.split('/')[-1].split('_')[1]
    A = np.load(f, allow_pickle=True)
    g = '/workspace/_selftest/old/lora_' + ctrl + '_' + str(seed) + '.npz'
    B = np.load(g, allow_pickle=True)
    diffs = []
    for k in ('pre_14d', 'post_14d', 'applied_canonical', 'eef_xyz', 'opening'):
        if not np.allclose(A[k], B[k]):
            diffs.append((k, float(np.abs(A[k] - B[k]).max())))
    pre = np.asarray(A['pre_14d'], float)
    pw = np.asarray(A['post_wrap_14d'], float)
    po = np.asarray(A['post_14d'], float)
    wr = round(100 * float(np.any(np.abs(pw - pre) > 1e-9, axis=1).mean()), 2)
    sf = round(100 * float(np.any(np.abs(po - pw) > 1e-9, axis=1).mean()), 2)
    oracle = int(np.asarray(A['oracle_cookie']).max())
    has_cookie = 'cookie_xyz' in A
    shape = A['cookie_xyz'].shape if has_cookie else None
    aligned = bool(has_cookie and np.array_equal(A['cookie_step'], np.arange(len(A['cookie_xyz']))))
    print('CTRL', ctrl, 'seed', seed, 'equiv_old_vs_new', (not diffs), diffs,
          'wrapper_pct', wr, 'safety_pct', sf, 'oracle_cookie_reads', oracle,
          'cookie_logged', has_cookie, shape, 'aligned', aligned)
print('SELFTEST_DONE')
