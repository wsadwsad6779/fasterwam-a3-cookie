import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
import unittest

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
import numpy as np

ROOT = Path('/workspace/eval_restore_20261117')
DEPLOY = ROOT / 'deploy'
report = {'gpu_execution': False, 'checks': {}, 'errors': [], 'original_stats_found': False}

def check(name, fn):
    try:
        report['checks'][name] = {'pass': True, 'facts': fn()}
    except Exception as e:
        report['checks'][name] = {'pass': False, 'error': type(e).__name__ + ': ' + str(e)}


def code_integrity():
    manifest = json.loads((ROOT / 'transfer_manifest.json').read_text())
    for entry in manifest:
        data = (DEPLOY / entry['path']).read_bytes()
        assert len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
        if entry['path'].endswith('.py'):
            ast.parse(data, filename=entry['path'])
    node = json.loads((ROOT / 'step_040000/manifest.json').read_text())
    expected = node.get('code_sha256')
    if expected is None:
        raise ValueError('manifest keys: ' + str(list(node)))
    for rel, sha in expected.items():
        path = DEPLOY / 'stage_40k' / rel
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, rel
    return {'transferred_files': len(manifest), 'node_code_matches': len(expected)}


def derived_stats():
    src = ROOT / 'archive_evidence/data'
    base = np.load(src / 'a3_fw/arrays.npz', allow_pickle=False)
    force = np.load(src / 'a3_fw_force/arrays.npz', allow_pickle=False)
    stats = json.loads((src / 'a3_fw/dataset_stats.json').read_text())
    fs = json.loads((src / 'a3_fw_force/force_stats.json').read_text())
    calib = json.loads((ROOT / 'step_040000/force_calib_v1.json').read_text())
    raw = force['force18'][:, calib['channels']].astype(np.float64)
    zero = np.asarray(calib['zero_load_ref_median'])
    residual = raw - zero
    std = residual.std(axis=0)
    np.testing.assert_allclose(std, calib['std'], rtol=1e-12, atol=1e-12)
    for key in ('action', 'opening', 'ep_start', 'ep_len', 'joint_cmd'):
        np.testing.assert_array_equal(base[key], force[key])
    assert base['proprio'].shape == (42681, 8)
    state8 = stats['state']['default']
    std32 = std.astype(np.float32).astype(np.float64)
    tail = {'global_min': -std32, 'global_max': std32,
            'global_mean': residual.mean(axis=0), 'global_std': std,
            'global_q01': np.percentile(residual, 1, axis=0),
            'global_q99': np.percentile(residual, 99, axis=0)}
    result = {'action': stats['action'], 'state': {'default': {
        key: list(state8[key]) + value.tolist() for key, value in tail.items()}}}
    dest = ROOT / 'derived/dataset_stats_14d_DERIVED.json'
    dest.parent.mkdir(exist_ok=True)
    with dest.open('x') as f:
        json.dump(result, f, indent=2)
    provenance = {'status': 'DERIVED_NOT_ORIGINAL', 'algorithm': 'stage_40k/build_v1_dataset.py global, lines 116-149 and 191-192',
                  'sources': {}, 'original_stats_sha256': None,
                  'training_corpus_sha256': calib['source_sha256']['corpus_arrays'],
                  'training_corpus_reconstructed': False,
                  'limitations': ['Original 14D JSON bytes/hash unavailable.', 'Derived values do not by themselves prove original training input identity.'],
                  'derived_sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
    for path in [src / 'a3_fw/arrays.npz', src / 'a3_fw/dataset_stats.json', src / 'a3_fw_force/arrays.npz', src / 'a3_fw_force/force_stats.json', ROOT / 'step_040000/force_calib_v1.json']:
        provenance['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (dest.parent / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    return {'path': str(dest), 'sha256': provenance['derived_sha256'], 'status': provenance['status'], 'widths': [14, 7], 'calibration_std_match': True, 'force_stats_keys': list(fs)}


def normalizer_cpu():
    import torch
    path = DEPLOY / 'stage_40k/fw/src/fasterwam/datasets/lerobot/utils/normalizer.py'
    spec = importlib.util.spec_from_file_location('restored_normalizer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = module.load_dataset_stats_from_json(str(ROOT / 'derived/dataset_stats_14d_DERIVED.json'))
    norm = module.LinearNormalizer(shape_meta={'action': [{'key': 'default'}], 'state': [{'key': 'default'}]}, use_stepwise_action_norm=False, default_mode='min/max', exception_mode=None, stats=stats)
    x = torch.zeros((1, 14))
    a = torch.tensor([[.1, -.2, .3, -.4, .5, -.6, 1.]])
    result = norm.forward({'state': {'default': x}, 'action': {'default': a.clone()}})
    assert result['state']['default'].shape == (1, 14)
    assert torch.isfinite(result['state']['default']).all()
    restored = norm.backward(result)
    assert torch.allclose(restored['action']['default'], a)
    return {'original_normalizer_source': True, 'derived_stats_only': True, 'torch': torch.__version__}


def capture_tests():
    path = DEPLOY / 'capture/test_evaluation_capture.py'
    source = path.read_text()
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ORIGINAL' for t in node.targets):
            node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()), args=[ast.Constant(str(DEPLOY / 'stage_40k/dev/eval40k'))], keywords=[])
    ast.fix_missing_locations(tree)
    sys.path.insert(0, str(path.parent))
    scope = {'__file__': str(path), '__name__': 'restored_capture_tests'}
    exec(compile(tree, str(path), 'exec'), scope)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(scope['CaptureTests']) if 'CaptureTests' in scope else unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(v) for v in scope.values() if isinstance(v, type) and issubclass(v, unittest.TestCase) and v is not unittest.TestCase)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful()
    return {'tests': result.testsRun, 'skipped': len(result.skipped), 'source_path_binding': 'AST-only; original file unchanged'}


def sim_resources():
    import mujoco
    sim = DEPLOY / 'a3_dual_arm_sim'
    xml = sim / 'models/a3_cookie_transfer.xml'
    model = mujoco.MjModel.from_xml_path(str(xml))
    sensors = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in range(model.nsensor)]
    return {'xml_sha256': hashlib.sha256(xml.read_bytes()).hexdigest(), 'sensors': sensors, 'nsensordata': model.nsensordata, 'nq': model.nq, 'mujoco': mujoco.__version__, 'episode_run': False}

check('code_integrity', code_integrity)
check('derived_stats', derived_stats)
check('normalizer_cpu', normalizer_cpu)
check('capture_tests', capture_tests)
check('sim_resources', sim_resources)
report['inference_identity_accepted'] = False
report['blocking'] = ['Original 14D stats file identity not recovered; derived stats are quarantined and not silently substituted.', 'No GPU model construction, merge, prompt encoding or inference was performed.']
(ROOT / 'cpu_restore_acceptance.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
