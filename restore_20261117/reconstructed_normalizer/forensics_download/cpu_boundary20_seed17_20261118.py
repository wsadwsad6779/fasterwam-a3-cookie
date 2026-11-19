"""CPU-only extraction of existing seed17 boundary evidence; no model/simulator imports."""
from pathlib import Path
import ast
import hashlib
import json
import runpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'forensics_download'
POLICY = ROOT.parent.parent / 'stage_40k/dev/eval40k/policy40k.py'
STREAM = POLICY.with_name('force_stream.py')
OUTPUT = FOLDER / 'CPU_BOUNDARY20_SEED17_20261118.json'
IDENTITY = 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name):
    with np.load(FOLDER / name, allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in archive.files}


def clean(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def main():
    steps = load('seed17_steps.npz')
    chunks = load('seed17_original_chunks.npz')
    obs = load('cpu_seed17_state_force.npz')
    meta = json.loads((FOLDER / 'seed17_meta.json').read_text(encoding='utf-8'))
    for archive in (steps, chunks, obs):
        assert str(archive['NORMALIZER_IDENTITY']) == IDENTITY
        assert str(archive['original_stats_sha256']) == 'UNKNOWN'
    # Extract the actual local policy function without importing torch or loading a model.
    tree = ast.parse(POLICY.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'noise_seed')
    namespace = {'np': np, 'INT32_MAX': 2**31 - 1}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(POLICY), 'exec'), namespace)
    noise_seed = namespace['noise_seed']
    stream_module = runpy.run_path(str(STREAM))  # numpy-only, no model/simulator
    stream = stream_module['ForceStream']()
    channels = list(stream_module['CHANNELS'])
    histories = []
    observations = []
    for f in range(21):
        stream.push(obs['observation.force'][f])
        history, mask = stream.snapshot()
        valid = min(f + 1, 16)
        assert int(mask.sum()) == valid
        assert np.array_equal(history[-valid:], obs['observation.force'][max(0, f-15):f+1, channels])
        assert not history[:-valid].any()
        histories.append({'observation_frame': f, 'status': 'CPU_RECONSTRUCTED_FROM_RECORDED_FORCE18_NOT_ORIGINAL_CAPTURE',
                          'source_frames_newest_last': [None]*(16-valid) + list(range(max(0, f-15), f+1)),
                          'channels': channels, 'raw_history16x6': history, 'mask16': mask, 'valid_rows': valid})
        observations.append({'frame': f, 'time_s': obs['time'][f], 'state16': obs['observation.state'][f],
                             'opening_state7_normalized_measured_not_meters': obs['observation.state'][f, 7],
                             'force18': obs['observation.force'][f], 'eef_pose14': obs['observation.eef_pose'][f],
                             'velocity16': obs['observation.velocity'][f], 'safety_stop': obs['safety_stop'][f]})

    def plan(c, r):
        prefix = f'chunk_{c:03d}_'
        result = {'chunk': c, 'row': r, 'fields_source': 'seed17_original_chunks.npz'}
        for out, key in [('raw_model_out7', 'model_out'), ('denorm7', 'denorm'), ('pm1_stage7', 'pm1'), ('adapter14', 'adapter_out')]:
            if prefix + key in chunks:
                result[out] = chunks[prefix + key].reshape(-1, 14 if key == 'adapter_out' else 7)[r]
        return result

    execution = []
    for i in range(20):
        c, r = int(steps['executed_chunk'][i]), int(steps['executed_chunk_row'][i])
        assert (c, r) == divmod(i, 10)
        row = plan(c, r)
        row.update(executed=True, control_step_zero_based=i, observation_before_frame=i,
                   observation_after_frame=int(steps['frame'][i]), opening_before=obs['observation.state'][i, 7],
                   opening_after=obs['observation.state'][i+1, 7])
        for field in ['handed_14d', 'requested_14d', 'applied_joint_16d', 'ik_joint_target_16d', 'ik_last_applied_joint_16d', 'proprio14']:
            if field in steps:
                row[field] = steps[field][i]
        execution.append(row)
    old = plan(0, 10)
    old.update(executed=False, status='OLD_UNEXECUTED_PLAN_ONLY', hypothetical_control_step_zero_based=10,
               observation_after_frame=None, note='Not handed, not applied, no attributable observed outcome; do not copy new execution fields here.')
    boundary = {'previous_actual_chunk0_row9': execution[9], 'old_unexecuted_chunk0_row10': old,
                'new_actual_chunk1_row0': execution[10]}
    noises = []
    for c in range(2):
        calculated = noise_seed(17, c+1)
        recorded = meta['chunks'][c]['meta']['noise_seed']
        assert calculated == recorded
        noises.append({'chunk': c, 'call_one_based': c+1, 'observation_frame': c*10,
                       'policy40k_noise_seed_recomputed': calculated, 'original_meta_noise_seed': recorded,
                       'equal': True, 'recorded_history_mask_true': meta['chunks'][c]['meta']['history_mask_true']})
    raw = np.array([r['raw_model_out7'][6] for r in execution])
    checks = {
        'frame_mapping_1_to_20': np.array_equal(steps['frame'][:20], np.arange(1, 21)),
        'handed_equals_adapter_first20': all(np.array_equal(r['handed_14d'], r['adapter14']) for r in execution),
        'handed_equals_requested_first20': np.array_equal(steps['handed_14d'][:20], steps['requested_14d'][:20]),
        'observed_opening_matches_steps': np.allclose(obs['observation.state'][1:21, 7], steps['opening_after'][:20], atol=1e-6, rtol=0),
        'pm1_gripper_equals_sign_raw': all(r['pm1_stage7'][6] == (1 if r['raw_model_out7'][6] >= 0 else -1) for r in execution),
        'adapter_gripper_is_negative_pm1': all(r['adapter14'][6] == -r['pm1_stage7'][6] for r in execution),
        'frame10_has_11_valid_history_rows': histories[10]['valid_rows'] == 11,
    }
    assert all(checks.values())
    summary = {'boundary_raw_gripper_previous_oldplan_new': [execution[9]['raw_model_out7'][6], old['raw_model_out7'][6], execution[10]['raw_model_out7'][6]],
               'boundary_pm1_gripper_previous_oldplan_new': [execution[9]['pm1_stage7'][6], old['pm1_stage7'][6], execution[10]['pm1_stage7'][6]],
               'boundary_adapter_gripper_previous_oldplan_new': [execution[9]['adapter14'][6], old['adapter14'][6], execution[10]['adapter14'][6]],
               'raw_gripper_first20': raw, 'raw_gripper_sign_flip_control_steps_zero_based': np.flatnonzero((raw[1:] >= 0) != (raw[:-1] >= 0))+1,
               'opening_normalized_frames_0_9_10_11_20': obs['observation.state'][[0, 9, 10, 11, 20], 7]}
    files = [FOLDER / n for n in ['seed17_steps.npz', 'seed17_original_chunks.npz', 'cpu_seed17_state_force.npz', 'seed17_meta.json']] + [POLICY, STREAM]
    result = {'NORMALIZER_IDENTITY': IDENTITY, 'original_stats_sha256': 'UNKNOWN', 'seed': 17,
              'scope': 'CPU local existing evidence only; no GPU, network, model inference or simulator rollout',
              'index_contract': 'step index i executes chunk=i//10 row=i%10 after observation frame i, yielding observation frame i+1; chunk0 row10 is never executed',
              'field_semantics': {'raw_model_out7': 'Original model_out, pre-denormalization; not physical units',
                                  'denorm7': 'Original separately recorded denorm array',
                                  'pm1_stage7': 'Original pm1 array: denormalized first6 and rewritten gripper',
                                  'opening': 'Measured normalized state[7], NOT meters',
                                  'force18': 'Recorded full observation.force; mixed channels, do not treat all18 norm as newtons',
                                  'eef_pose14': 'Recorded observation.eef_pose, no pose synthesized'},
              'sources': [{'absolute_path': str(p.resolve()), 'sha256': sha(p)} for p in files],
              'archive_schema': {name: {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in a.items()} for name, a in [('seed17_steps.npz', steps), ('seed17_original_chunks.npz', chunks), ('cpu_seed17_state_force.npz', obs)]},
              'missing_original_fields': [
                  'No directly captured original per-call force_history16x6 or mask16 in these three NPZs; histories here are explicitly CPU reconstructions; original metadata only records valid count.',
                  'No original per-call normalized proprio tensor or normalized force_history tensor in these three NPZs; steps.proprio14 is retained only under its real name, not relabeled normalized model input.',
                  'No original full diffusion noise/latent tensors or RNG snapshots in these input NPZs; noise_seed comes from original meta and independently recomputed policy function.',
                  'No handed/requested/applied/observed outcome for old chunk0 row10: it was not executed; recorded adapter_out row10 is only a plan.',
                  'Original training normalizer/statistics unavailable: original_stats_sha256 UNKNOWN.',
                  'No physical meter-valued gripper opening field; measured state[7] is normalized.'
              ],
              'noise_seed_function': {'path': str(POLICY.resolve()), 'function_source': ast.get_source_segment(POLICY.read_text(encoding='utf-8'), fn), 'method': 'AST extract function only, no policy module/model import'},
              'noise_seeds': noises, 'boundary': boundary, 'executed_first20': execution,
              'observations_frames0_to20': observations, 'histories_frames0_to20': histories,
              'checks': checks, 'summary': summary}
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=clean, allow_nan=False), encoding='utf-8')
    print(json.dumps({'output': str(OUTPUT), 'summary': summary, 'noise_seeds': noises, 'checks': checks}, default=clean, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
