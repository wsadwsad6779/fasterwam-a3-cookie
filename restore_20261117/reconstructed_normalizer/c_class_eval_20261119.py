"""C-class acceptance: independent bounded-LoRA checkpoint, bare policy, no external gates.

System split this script measures:
  A = original step_040000_merged.pt, bare protocol (same loop, same seeds)
  C = step_000100.pt produced by the bounded 300-step LoRA trial (stopped at 100 by disk
      exhaustion during the step-200 save; the 200 file was verified unloadable)

C must be judged with deployable inputs ONLY: images, 14-dim proprio, the frozen 6-channel
force stream, and text. No G3 release gate, no M5 safety controller, no cookie world
coordinates, no oracle decisions. The rollout loop is the archived bare loop
(rollout40k.run_episode) copied verbatim except for one added passive safety abort
(wrist > 150 N or pad > 500 N in a single frame), which applies to BOTH arms.

Seeds 29/30 are the frozen held-out test seeds from MODEL_REPAIR_TASKS.json.

Markers on every artefact:
  NORMALIZER_IDENTITY=RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL
  TRAINING_DATA_IDENTITY=CANDIDATE_RECONSTRUCTED_TRAINING_DATA
  TEACHER_DATA_IDENTITY=CANDIDATE_TEACHER_DATA
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('C_OUT', 'c_class_eval_20261119')
RESUME = os.environ.get('C_RESUME', '0') == '1'
if OUT.exists() and not RESUME:
    raise SystemExit(f'{OUT} exists; refusing to overwrite previous C-class evidence')
if not OUT.exists():
    OUT.mkdir(parents=True)

CKPT = Path(os.environ.get(
    'C_CKPT', str(ROOT / 'model_repair_runs' / 'lora_300' / 'step_000100.pt')))
SEEDS = [int(s) for s in os.environ.get('C_SEEDS', '29,30').split(',')]
MAX_STEPS = int(os.environ.get('C_MAX_STEPS', '600'))
REPLAN = int(os.environ.get('C_REPLAN', '10'))
WRIST_ABORT_N = float(os.environ.get('C_WRIST_ABORT_N', '150'))
PAD_ABORT_N = float(os.environ.get('C_PAD_ABORT_N', '500'))
VIDEO = os.environ.get('C_VIDEO', '1') == '1'
TRAINABLE_TAG = 'action_expert'

MARKERS = {
    **D.MARKERS,
    'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
    'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA',
    'CHECKPOINT_IDENTITY': 'ACTION_EXPERT_LOCAL_FULL_FINETUNE (not LoRA)',
    'EVAL_GATES': 'NONE (no G3 release gate, no M5 controller, no oracle inputs)',
}

E, factory, W, fw = D.setup_protocol()
import torch
from a3_dual_arm_sim.contracts import STATE
from force_stream import CHANNELS, CHANNEL_NAMES, FORCE_OBS_KEY, select_channels
from rollout40k import _success_of


def write(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')


def build_policy():
    policy, doc, calib, facts = D.build_policy_reconstructed(
        E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
    assert facts['device'] == 'cuda' and not policy.model.training
    assert list(calib['channels']) == list(CHANNELS)
    return policy, doc, calib, facts


SAMPLE_KEYS_HINT = ('action_expert', 'blocks.0.', 'text_encoder', 'proprio')


def load_ckpt(policy, path):
    """Load a full-state or delta-only checkpoint and prove the change stays on the whitelist."""
    sd = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    kind = 'DELTA_ONLY' if 'delta' in sd else 'FULL_STATE_DICT'
    ck = sd['delta'] if kind == 'DELTA_ONLY' else sd['model']
    own = policy.model.state_dict()
    if kind == 'DELTA_ONLY':
        unexpected = [k for k in ck if k not in own]
        shape_bad = [k for k in ck if k in own and tuple(own[k].shape) != tuple(ck[k].shape)]
        missing = []
    else:
        missing = [k for k in own if k not in ck]
        unexpected = [k for k in ck if k not in own]
        shape_bad = [k for k in own if k in ck and tuple(own[k].shape) != tuple(ck[k].shape)]
    if missing or unexpected or shape_bad:
        raise SystemExit('%s does not match the eval model: missing=%d unexpected=%d '
                         'shape=%d first=%s|%s|%s' % (kind, len(missing), len(unexpected),
                                                     len(shape_bad), missing[:3],
                                                     unexpected[:3], shape_bad[:3]))

    # Sample the base weights before loading: only the whitelisted tensors may change.
    tr = [k for k in own if TRAINABLE_TAG in k]
    ot = [k for k in own if TRAINABLE_TAG not in k]
    sample = tr[::max(1, len(tr) // 10)][:10] + ot[::max(1, len(ot) // 10)][:10]
    base = {k: own[k].detach().to('cpu', copy=True) for k in sample}
    del own

    policy.model.load_state_dict({k: v for k, v in ck.items()},
                                 strict=(kind == 'FULL_STATE_DICT'))
    policy.model.eval()
    now = policy.model.state_dict()
    diffs = {}
    for k in sample:
        nb = now[k].detach().to('cpu')
        diffs[k] = {'max_abs_diff': float((base[k].float() - nb.float()).abs().max()),
                    'bit_equal': bool(torch.equal(base[k], nb.to(base[k].dtype))),
                    'trainable': TRAINABLE_TAG in k}
    del base
    tr_diff = [v['max_abs_diff'] for k, v in diffs.items() if v['trainable']]
    ot_diff = [v['max_abs_diff'] for k, v in diffs.items() if not v['trainable']]
    return {'checkpoint': str(path), 'kind': kind, 'step': int(sd.get('step', -1)),
            'trainable_tensors_saved': len(sd.get('trainable', [])),
            'keys': len(ck), 'delta_confined_to_whitelist':
                bool(ot_diff and max(ot_diff) == 0.0 and tr_diff and max(tr_diff) > 0),
            'sample_trainable_max_abs_diff': max(tr_diff) if tr_diff else None,
            'sample_frozen_max_abs_diff': max(ot_diff) if ot_diff else None,
            'sample': diffs}


def snap(store, obs, info, action=None):
    store['eef'].append(np.asarray(obs['observation.eef_pose'], dtype=np.float64))
    store['force18'].append(np.asarray(obs[FORCE_OBS_KEY], dtype=np.float64))
    store['opening'].append(float(np.asarray(obs[STATE])[7]))
    if action is not None:
        store['action'].append(np.asarray(action, dtype=np.float64))
    occ = info.get('target_slot_occupancy')
    store['cookies'].append(-1 if info.get('cookies_in_target') is None
                            else int(info['cookies_in_target']))
    store['source'].append(-1 if info.get('cookies_in_source') is None
                           else int(info['cookies_in_source']))
    store['exact'].append(-1 if not info else int(bool(info.get('exact_2x5_fill'))))
    store['touches'].append(-1 if not info else int(bool(info.get('target_touches_all_walls'))))
    store['hold'].append(-1 if not info else int(info.get('success_hold_count') or 0))
    store['success'].append(-1 if not info or info.get('success') is None
                            else int(bool(info['success'])))
    store['occ'].append([-1] * 10 if occ is None else [int(v) for v in occ])


def frame_img(obs):
    from PIL import Image

    def r(a):
        a = np.asarray(a)
        if a.shape[0] == 224 and a.shape[1] == 224:
            return a.astype(np.uint8)
        return np.asarray(Image.fromarray(a).resize((224, 224), Image.BILINEAR), np.uint8)
    return np.concatenate([r(obs['observation.images.front']),
                           r(obs['observation.images.left_wrist'])], axis=1)


def rollout(env, policy, seed, reset):
    """Bare loop copied from rollout40k.run_episode + passive high-force abort."""
    obs = reset(env, seed)
    policy.start_episode(obs, seed)
    store = {k: [] for k in ('eef', 'force18', 'opening', 'action', 'cookies', 'source',
                             'exact', 'touches', 'hold', 'success', 'occ')}
    snap(store, obs, {})
    vframes = [frame_img(obs)] if VIDEO else None
    measured = [select_channels(obs[FORCE_OBS_KEY], where='frame 0')]
    policy.observe(obs, 0)
    chunks = []
    pending = []
    info = {}
    nan_abort = -1
    abort = None
    terminated = False
    truncated = False
    t0 = time.time()
    step = -1
    for step in range(MAX_STEPS):
        if not pending:
            actions, pred_force, meta = policy.chunk(obs, frame=len(measured) - 1, seed=seed)
            take = [np.asarray(a, dtype=np.float64) for a in actions[:REPLAN]]
            pending.extend(take)
            chunks.append({'frame': len(measured) - 1,
                           'pred_force': None if pred_force is None
                           else np.asarray(pred_force, dtype=np.float64).tolist(),
                           'n_actions': len(take), 'meta': meta})
            print('[seed %d] chunk frame=%d t=%.1fs' % (seed, len(measured) - 1,
                                                        time.time() - t0), flush=True)
        action = pending.pop(0)
        obs, _reward, terminated, truncated, info = env.step(action)
        if VIDEO:
            vframes.append(frame_img(obs))
        frame = len(measured)
        f18 = np.asarray(obs[FORCE_OBS_KEY], dtype=np.float64)
        if not np.all(np.isfinite(f18)):
            nan_abort = frame
            break
        measured.append(select_channels(f18, where='frame %d' % frame))
        policy.observe(obs, frame)
        snap(store, obs, info, action)
        m6 = measured[-1]
        if np.abs(m6[:3]).max() > WRIST_ABORT_N or m6[3:5].max() > PAD_ABORT_N:
            abort = {'frame': frame, 'reason': 'force_abort', 'measured': m6.tolist()}
            break
        if terminated or truncated:
            break
    success, in_target = _success_of(info)
    record = {'seed': int(seed), 'frames': len(measured),
              'steps_executed': len(measured) - 1,
              'decision_chunks': len(chunks),
              'terminated': bool(terminated), 'truncated': bool(truncated),
              'success': success,
              'cookies_in_target': None if in_target is None else int(in_target),
              'nan_abort_frame': int(nan_abort), 'safety_abort': abort,
              'wall_s': round(time.time() - t0, 2)}
    return record, measured, chunks, store, vframes


def save_video(arm, seed, vframes):
    import cv2
    h, w = vframes[0].shape[:2]
    p = OUT / ('episode_%s_seed%d.mp4' % (arm, seed))
    wr = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (w, h))
    if not wr.isOpened():
        raise SystemExit('video writer refused %s' % p)
    for f in vframes:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()
    return {'path': str(p), 'frames': len(vframes), 'bytes': p.stat().st_size,
            'fps': 20, 'backend': 'cv2/mp4v'}


def episode_metrics(record, measured, chunks, store):
    m = np.asarray(measured, dtype=np.float64)
    eef = np.asarray(store['eef'], dtype=np.float64)
    act = np.asarray(store['action'], dtype=np.float64) if store['action'] else np.zeros((0, 7))
    disp = (np.linalg.norm(eef[1:, :3] - eef[:-1, :3], axis=1) * 1000.0
            if len(eef) > 1 else np.zeros(0))
    cmd = np.abs(act[:len(disp), :3]).max(1) * 25.0 if len(disp) else np.zeros(0)
    track = np.divide(disp, cmd, out=np.ones_like(disp), where=cmd > 0)
    jam = int(((cmd > 20.0) & (track < 0.3)).sum())
    gi = act[:, 6] if act.size else np.zeros(0)
    flips = int((np.sign(gi[1:]) != np.sign(gi[:-1])).sum()) if len(gi) > 1 else 0
    cookies = np.asarray(store['cookies'])
    source = np.asarray(store['source'])
    exact = np.asarray(store['exact'])
    src = source[source >= 0]
    grasps = int((np.diff(src) < 0).sum()) if len(src) > 1 else 0
    removed = int(src[0] - src[-1]) if len(src) else 0
    close_ev = int(((np.sign(gi[1:]) > 0) & (np.sign(gi[:-1]) <= 0)).sum()) if len(gi) > 1 else 0
    return {
        **{k: v for k, v in MARKERS.items()},
        'record': record,
        'cookies_final': int(cookies[-1]), 'cookies_max': int(cookies.max()),
        'cookies_frames_with_any': int((cookies > 0).sum()),
        'source_final': int(source[-1]), 'source_min': int(source[source >= 0].min()),
        'exact_fill_any': bool((exact > 0).any()),
        'exact_fill_frames': int((exact > 0).sum()),
        'success_frames': int((np.asarray(store['success']) > 0).sum()),
        'occupancy_final': store['occ'][-1],
        'touches_all_walls_final': int(store['touches'][-1]),
        'hold_final': int(store['hold'][-1]),
        'force_envelope': {'absmax': float(np.abs(m).max()),
                           'per_channel_max': m.max(0).tolist(),
                           'frames_gt_50': int((np.abs(m) > 50).any(1).sum()),
                           'frames_gt_14_3': int((np.abs(m) > 14.3).any(1).sum())},
        'eef_path_mm': float(disp.sum()), 'eef_step_mm_max': float(disp.max()) if len(disp) else 0.0,
        'jam_steps': jam, 'gripper_flips': flips,
        'gripper_close_events': close_ev, 'grasps_confirmed': grasps,
        'cookies_removed_from_source': removed,
        'grasp_success_rate': (grasps / close_ev) if close_ev else None,
        'opening_min': float(np.min(store['opening'])), 'opening_max': float(np.max(store['opening'])),
        'chunks': [{'frame': c['frame'], 'n_actions': c['n_actions'],
                    'meta': c['meta'],
                    'pred_force_absmax': None if c['pred_force'] is None
                    else float(np.abs(np.asarray(c['pred_force'])).max())} for c in chunks],
        'infer_calls': len(chunks),
    }


def save_episode(arm, seed, record, measured, chunks, store, metrics):
    np.savez_compressed(
        OUT / ('episode_%s_seed%d.npz' % (arm, seed)),
        measured=np.asarray(measured, dtype=np.float64),
        eef=np.asarray(store['eef'], dtype=np.float64),
        force18=np.asarray(store['force18'], dtype=np.float64),
        opening=np.asarray(store['opening'], dtype=np.float64),
        action=np.asarray(store['action'], dtype=np.float64) if store['action']
        else np.zeros((0, 7)),
        cookies=np.asarray(store['cookies'], dtype=np.int64),
        source=np.asarray(store['source'], dtype=np.int64),
        exact=np.asarray(store['exact'], dtype=np.int64),
        touches=np.asarray(store['touches'], dtype=np.int64),
        hold=np.asarray(store['hold'], dtype=np.int64),
        success=np.asarray(store['success'], dtype=np.int64),
        occupancy=np.asarray(store['occ'], dtype=np.int64),
        chunk_frames=np.asarray([c['frame'] for c in chunks], dtype=np.int64),
        noise_seeds=np.asarray([int(c['meta']['noise_seed']) for c in chunks], dtype=np.int64))
    write('episode_%s_seed%d.json' % (arm, seed), metrics)


def run_arm(arm):
    policy, doc, calib, facts = build_policy()
    load_info = None
    if arm == 'C':
        load_info = load_ckpt(policy, CKPT)
        print('[C] checkpoint load ' + json.dumps(
            {k: v for k, v in load_info.items() if k != 'sample'}), flush=True)
        sha_file = OUT / 'checkpoint_sha256.txt'
        if sha_file.exists():
            load_info['sha256'] = sha_file.read_text().strip()
    results = []
    for seed in SEEDS:
        env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
        reset = D.make_reset(E, factory, env, SimpleNamespace(
            cpu=False, gripper_mode='pm1', max_steps=MAX_STEPS, replan=REPLAN,
            allow_invalid_scene=False))
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        record, measured, chunks, store, vframes = rollout(env, policy, seed, reset)
        if hasattr(env, 'close'):
            env.close()
        metrics = episode_metrics(record, measured, chunks, store)
        if VIDEO:
            metrics['video'] = save_video(arm, seed, vframes)
            print('[%s seed %d] video %s frames=%d %.1fMB' % (
                arm, seed, metrics['video']['path'], metrics['video']['frames'],
                metrics['video']['bytes'] / 1e6), flush=True)
        save_episode(arm, seed, record, measured, chunks, store, metrics)
        results.append(metrics)
        print('[%s seed %d] success=%s target=%s(max %s) source=%s exact=%s force_absmax=%.2f '
              'jam=%d grasps=%d/%d wall=%.1fs' % (arm, seed, record['success'], metrics['cookies_final'],
                                     metrics['cookies_max'], metrics['source_final'],
                                     metrics['exact_fill_any'], metrics['force_envelope']['absmax'],
                                     metrics['jam_steps'], metrics['grasps_confirmed'],
                                     metrics['gripper_close_events'], record['wall_s']), flush=True)
    del policy
    torch.cuda.empty_cache()
    return {'arm': arm, 'facts': facts, 'load': load_info, 'episodes': results}


def paired(a, c):
    out = {'per_seed': []}
    for ea in a['episodes']:
        seed = ea['record']['seed']
        ec = next(x for x in c['episodes'] if x['record']['seed'] == seed)
        out['per_seed'].append({
            'seed': seed,
            'A': {k: ea.get(k) for k in ('cookies_final', 'cookies_max', 'source_final',
                                         'exact_fill_any', 'force_envelope', 'jam_steps',
                                         'eef_path_mm', 'success_frames', 'grasps_confirmed',
                                         'gripper_close_events', 'grasp_success_rate')},
            'C': {k: ec.get(k) for k in ('cookies_final', 'cookies_max', 'source_final',
                                         'exact_fill_any', 'force_envelope', 'jam_steps',
                                         'eef_path_mm', 'success_frames', 'grasps_confirmed',
                                         'gripper_close_events', 'grasp_success_rate')},
            'delta_cookies': ec['cookies_final'] - ea['cookies_final'],
            'delta_force_absmax': (ec['force_envelope']['absmax']
                                   - ea['force_envelope']['absmax']),
        })
    out['C_beats_A_on_any_seed'] = any(p['delta_cookies'] > 0 for p in out['per_seed'])
    out['C_exact_fill_any_seed'] = any(e['exact_fill_any'] for e in c['episodes'])
    out['C_success_any_seed'] = any(e['record']['success'] for e in c['episodes'])
    return out


def main():
    if RESUME:
        a = json.loads((OUT / 'A_summary.json').read_text(encoding='utf-8'))
        print('[resume] A arm loaded from previous run: %d episodes'
              % len(a['episodes']), flush=True)
        c = run_arm('C')
        write('C_summary.json', {**MARKERS, **c})
        cmp = paired(a, c)
        write('paired_A_vs_C.json', {**MARKERS, **cmp})
        write('summary.json', {**MARKERS, 'complete': True, 'seeds': SEEDS, 'resumed': True,
                               'A': [{'seed': e['record']['seed'], 'success': e['record']['success'],
                                      'cookies_final': e['cookies_final'],
                                      'cookies_max': e['cookies_max'],
                                      'source_final': e['source_final'],
                                      'exact_fill_any': e['exact_fill_any'],
                                      'force_absmax': e['force_envelope']['absmax'],
                                      'jam_steps': e['jam_steps'],
                                      'safety_abort': e['record']['safety_abort']}
                                     for e in a['episodes']],
                               'C': [{'seed': e['record']['seed'], 'success': e['record']['success'],
                                      'cookies_final': e['cookies_final'],
                                      'cookies_max': e['cookies_max'],
                                      'source_final': e['source_final'],
                                      'exact_fill_any': e['exact_fill_any'],
                                      'force_absmax': e['force_envelope']['absmax'],
                                      'jam_steps': e['jam_steps'],
                                      'safety_abort': e['record']['safety_abort']}
                                     for e in c['episodes']],
                               'paired': cmp})
        print('C_CLASS_DONE', flush=True)
        return
    write('predeclared.json', {
        **MARKERS, 'seeds': SEEDS, 'max_steps': MAX_STEPS, 'replan': REPLAN,
        'arms': {'A': 'original step_040000_merged.pt, bare policy',
                 'C': 'step_000100.pt bounded LoRA checkpoint, bare policy'},
        'inputs': ['images', '14-dim proprio', '6-channel frozen force stream', 'text'],
        'forbidden_inputs': ['G3 release gate', 'M5 controller', 'cookie world coordinates',
                             'oracle decisions', 'simulator truth in the policy path'],
        'safety_abort': {'wrist_abs_max_gt_n': WRIST_ABORT_N, 'pad_abs_max_gt_n': PAD_ABORT_N,
                         'note': 'passive abort, identical for both arms, not a release gate'},
        'checkpoint': str(CKPT), 'checkpoint_exists': CKPT.exists(),
        'provenance': 'train_lora_300.py / train_action_expert_segmented_20261119.py: '
                      '602 action_expert tensors (full local fine-tune, no LoRA factors), '
                      'lr schedule documented in the trainer, seed 20261118',
        'training_crash': 'step 200 save failed with disk full (/workspace 98G); '
                          'step_000200.pt lacks a zip central directory and is not loadable; '
                          'step_000100.pt verified loadable (4746 keys, step=100)',
    })
    print('[env] gpu=' + torch.cuda.get_device_name(0), flush=True)
    a = run_arm('A')
    write('A_summary.json', {**MARKERS, **a})
    c = run_arm('C')
    write('C_summary.json', {**MARKERS, **c})
    cmp = paired(a, c)
    write('paired_A_vs_C.json', {**MARKERS, **cmp})
    write('summary.json', {**MARKERS, 'complete': True, 'seeds': SEEDS,
                           'A': [{'seed': e['record']['seed'], 'success': e['record']['success'],
                                  'cookies_final': e['cookies_final'],
                                  'cookies_max': e['cookies_max'],
                                  'source_final': e['source_final'],
                                  'exact_fill_any': e['exact_fill_any'],
                                  'force_absmax': e['force_envelope']['absmax'],
                                  'jam_steps': e['jam_steps'],
                                  'safety_abort': e['record']['safety_abort']}
                                 for e in a['episodes']],
                           'C': [{'seed': e['record']['seed'], 'success': e['record']['success'],
                                  'cookies_final': e['cookies_final'],
                                  'cookies_max': e['cookies_max'],
                                  'source_final': e['source_final'],
                                  'exact_fill_any': e['exact_fill_any'],
                                  'force_absmax': e['force_envelope']['absmax'],
                                  'jam_steps': e['jam_steps'],
                                  'safety_abort': e['record']['safety_abort']}
                                 for e in c['episodes']],
                           'paired': cmp})
    print('C_CLASS_DONE', flush=True)


if __name__ == '__main__':
    main()
