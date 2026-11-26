#!/usr/bin/env python3
"""P0.2 denoise runtime probe: empirically count velocity-net forwards / scheduler steps
for K=5/10/20 on the frozen F2 15k, plus dim6 numeric pipeline (norm vs exec)."""
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace')
import numpy as np
import torch
import _t7_eval as T

CK = '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt'
OUT = '/workspace/fwv1/p_items/DENOISE_RUNTIME_CONFIG_AUDIT.json'
SEED = 7500
NOISE = 12345


def main():
    if not os.path.exists(CK):
        raise FileNotFoundError('ckpt missing (fail-closed): ' + CK)
    model, film, state = T.build_model(CK)
    h = __import__('hashlib').sha256(T.PROMPT.encode()).hexdigest()
    cache = torch.load(os.path.join(T.CACHE_DIR, '%s.t5_len%d.wan22ti2v5b.pt' % (h, T.CONTEXT_LEN)), map_location='cpu')
    ctx = cache['context'][None].cuda(); ctx_mask = cache['mask'].bool()[None].cuda()
    stats = T.Stats()
    _, asc, aoff = stats.forward(np.zeros(7, np.float32), stats.amin, stats.amax)

    from a3_dual_arm_sim import A3CookieTransferEnv
    sm = json.load(open(T.SCENE_MANIFEST)); sc = {int(s['seed']): s for s in sm['scenes']}
    env = A3CookieTransferEnv(config=T.SIM_CONFIG, action_mode='cartesian_delta', render_mode='rgb_array', render_cameras=False)
    obs, _ = env.reset(seed=SEED, options=dict(sc[SEED]['options']))
    img = T.image_to_model(obs)
    prop_raw = np.asarray(obs['observation.state'], np.float64)[:8]
    prop, _, _ = stats.forward(prop_raw, stats.smin, stats.smax)
    ring = np.zeros((T.HIST, 4), np.float32)
    zfeat = np.clip((ring - stats.fmean) / stats.fstd * stats.factive, -5, 5).reshape(1, -1)
    state['z'] = None if film is None else torch.tensor(zfeat, dtype=torch.float32).cuda()

    sched = model.infer_action_scheduler
    sched_name = type(sched).__name__
    sched_info = {}
    # inspect schedule builder if present
    for meth in ('build_inference_schedule',):
        if hasattr(sched, meth):
            try:
                tm, dl = getattr(sched, meth)(num_inference_steps=5)
                sched_info['timesteps_len_K5'] = int(np.size(tm)) if hasattr(tm, '__len__') else 'scalar'
                sched_info['timesteps_first_last_K5'] = [float(np.asarray(tm).reshape(-1)[0]), float(np.asarray(tm).reshape(-1)[-1])] if hasattr(tm, '__len__') else None
                sched_info['deltas_len_K5'] = int(np.size(dl)) if hasattr(dl, '__len__') else 'scalar'
            except Exception as e:
                sched_info['build_err'] = str(e)
    # shift attr if any
    for a in ('shift', 'sigma_shift', 'num_train_timesteps'):
        if hasattr(sched, a):
            try: sched_info[a] = getattr(sched, a)
            except Exception: pass

    results = {}
    for K in (5, 10, 20):
        c = {'fwd': 0, 'step': 0}
        orig_step = sched.step
        ae = model.action_expert
        orig_fwd = ae.forward

        def step(*a, **k):
            c['step'] += 1
            return orig_step(*a, **k)

        def fwd(*a, **k):
            c['fwd'] += 1
            return orig_fwd(*a, **k)

        sched.step = step
        ae.forward = fwd
        t0 = time.time()
        try:
            with torch.no_grad():
                out = model.infer_action(prompt=None, context=ctx, context_mask=ctx_mask,
                                         input_image=img.cuda(), action_horizon=32, num_video_frames=T.T_VID,
                                         proprio=torch.tensor(prop, dtype=torch.float32)[None].cuda(),
                                         num_inference_steps=K, seed=NOISE)
            a = np.asarray(out['action'].float().cpu().numpy())
            shape = list(a.shape)
            an = a.reshape(-1, 7)
            dn = stats.backward(an, asc, aoff)
            results[K] = {'forward_calls': c['fwd'], 'scheduler_steps': c['step'],
                          'infer_ms': round((time.time() - t0) * 1000, 1), 'out_shape': shape,
                          'grip_norm_min': float(an[:, 6].min()), 'grip_norm_max': float(an[:, 6].max()),
                          'grip_exec_min': float(dn[:, 6].min()), 'grip_exec_max': float(dn[:, 6].max()),
                          'grip_exec_median': float(np.median(dn[:, 6])),
                          'grip_exec_lt030_frac': float((dn[:, 6] < 0.30).mean())}
        except Exception as e:
            results[K] = {'error': str(e)[:400]}
        finally:
            sched.step = orig_step
            ae.forward = orig_fwd

    od = {'artifact': 'DENOISE_RUNTIME_CONFIG_AUDIT',
          'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'checkpoint': CK, 'checkpoint_sha256': T.sha256_file(CK),
          'scheduler_class': sched_name, 'sched_info': sched_info,
          'env_denoise_in_eval': int(os.environ.get('T7_DENOISE', '5')),
          'probe': 'single infer_action from fixed start obs (seed %d, noise seed %d), K in {5,10,20}' % (SEED, NOISE),
          'note': 'forward_calls = action_expert.forward invocations during ONE infer_action; scheduler_steps = scheduler.step calls; confirms whether K really changes denoise iterations (not just an env var).',
          'results': results}
    json.dump(od, open(OUT, 'w'), indent=1)
    print(json.dumps({'scheduler': sched_name, 'sched_info': sched_info, 'results': results}))


if __name__ == '__main__':
    main()
