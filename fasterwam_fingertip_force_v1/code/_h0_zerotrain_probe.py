#!/usr/bin/env python3
"""H0 zero-training H-horizon probe.

Loads a frozen checkpoint (default F0 step_009000.pt) and calls infer_action at
H=32/48/64 from IDENTICAL start frames, then compares the trained region (0..31)
against the untrained region (32..H-1). NO training, NO weight change.
This is exploratory OOD inference: positions 32+ were never supervised.
"""
import os
import sys
import json
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')
sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
sys.path.insert(0, '/workspace')

import numpy as np
import torch
import _t7_eval as T  # deployed eval module at /workspace/_t7_eval.py (build_model/image_to_model/Stats/constants)

HORIZONS = [int(x) for x in os.environ.get('H0_HORIZONS', '32,48,64').split(',') if x.strip()]
SEEDS = [int(x) for x in os.environ.get('H0_SEEDS', '7500,7501,7502,7503,7504').split(',') if x.strip()]
CKPT = os.environ.get('H0_CKPT', '/workspace/fwv1/f0_v3_baseline/step_009000.pt')
OUT = os.environ.get('H0_OUT', '/workspace/fwv1/p_items/H0_ZEROTRAIN_PROBE.json')
NPZ = os.environ.get('H0_NPZ', '/workspace/fwv1/p_items/H0_CHUNKS.npz')


def main():
    if CKPT and not os.path.exists(CKPT):
        raise FileNotFoundError('H0 checkpoint not found (fail-closed): %s' % CKPT)
    sm = json.load(open(T.SCENE_MANIFEST))
    scene_map = {int(s['seed']): s for s in sm['scenes']}
    stats = T.Stats()
    _, ascale, aoff = stats.forward(np.zeros(7, np.float32), stats.amin, stats.amax)

    h = __import__('hashlib').sha256(T.PROMPT.encode()).hexdigest()
    cache = torch.load(os.path.join(T.CACHE_DIR, '%s.t5_len%d.wan22ti2v5b.pt' % (h, T.CONTEXT_LEN)),
                       map_location='cpu')
    ctx = cache['context'][None].cuda()
    ctx_mask = cache['mask'].bool()[None].cuda()

    model, film, state = T.build_model(CKPT)

    from a3_dual_arm_sim import A3CookieTransferEnv
    env = A3CookieTransferEnv(config=T.SIM_CONFIG, action_mode='cartesian_delta',
                              render_mode='rgb_array', render_cameras=True)

    def denorm(a_norm):
        return stats.backward(a_norm, ascale, aoff)

    def infer_at(img, prop, zfeat, H, seed):
        state['z'] = None if (film is None or zfeat is None) else torch.tensor(zfeat, dtype=torch.float32).cuda()
        with torch.no_grad():
            out = model.infer_action(prompt=None, context=ctx, context_mask=ctx_mask,
                                     input_image=img.cuda(), action_horizon=int(H),
                                     num_video_frames=T.T_VID,
                                     proprio=torch.tensor(prop, dtype=torch.float32)[None].cuda(),
                                     num_inference_steps=T.DENOISE,
                                     seed=int(seed) * 100000 + 0)
        a = np.asarray(out['action'].float().cpu().numpy())
        if a.ndim == 3:
            a = a[0]
        if a.ndim != 2 or a.shape[0] != H:
            raise ValueError('unexpected action shape %s for H=%d' % (a.shape, H))
        return a

    def stats_block(a):
        # a: [H,7] denormed. trained = [0:32], tail = [32:]
        out = {}
        grip = a[:, 6]
        out['nan_count'] = int(np.isnan(a).sum())
        out['absmean'] = float(np.abs(a).mean())
        out['grip_lt_0p30_frac'] = float((grip < 0.30).mean())
        out['grip_gt_0p90_frac'] = float((grip > 0.90).mean())
        out['grip_median'] = float(np.median(grip))
        out['grip_min'] = float(grip.min())
        out['grip_max'] = float(grip.max())
        # jerk = second difference magnitude, per-step
        def jerk(x):
            if x.shape[0] < 3:
                return 0.0
            d2 = np.diff(np.diff(x, axis=0), axis=0)
            return float(np.abs(d2).mean())
        if a.shape[0] > 32:
            tr = a[:32]; tl = a[32:]
            out['trained_absmean'] = float(np.abs(tr).mean())
            out['tail_absmean'] = float(np.abs(tl).mean())
            out['trained_jerk'] = jerk(tr)
            out['tail_jerk'] = jerk(tl)
            out['tail_over_trained_jerk_ratio'] = (out['tail_jerk'] / out['trained_jerk']) if out['trained_jerk'] > 1e-9 else None
            out['boundary_jump_31_32'] = float(np.abs(a[32] - a[31]).max())
            out['tail_grip_lt_0p30_frac'] = float((tl[:, 6] < 0.30).mean())
        return out

    results = []
    chunks = {}
    for seed in SEEDS:
        t0 = time.time()
        try:
            obs, _ = env.reset(seed=seed, options=dict(scene_map[seed]['options']))
            img = T.image_to_model(obs)
            prop_raw = np.asarray(obs['observation.state'], np.float64)[:8]
            prop, _, _ = stats.forward(prop_raw, stats.smin, stats.smax)
            ring = np.zeros((T.HIST, 4), np.float32)
            zfeat = np.clip((ring - stats.fmean) / stats.fstd * stats.factive, -5, 5).reshape(1, -1)
            per_h = {}
            for H in HORIZONS:
                a = denorm(infer_at(img, prop, zfeat, H, seed))
                chunks['s%d_H%d' % (seed, H)] = a
                per_h[str(H)] = stats_block(a)
            # reproducibility: same state, H=32 twice
            a32a = denorm(infer_at(img, prop, zfeat, 32, seed))
            a32b = denorm(infer_at(img, prop, zfeat, 32, seed))
            repro = float(np.max(np.abs(a32a - a32b)))
            # trained-region stability across H: max abs diff of positions 0..31 vs H=32
            first32 = {}
            if 32 in HORIZONS:
                ref = chunks['s%d_H32' % seed][:32]
                for H in HORIZONS:
                    if H >= 32:
                        first32[str(H)] = float(np.max(np.abs(chunks['s%d_H%d' % (seed, H)][:32] - ref)))
            results.append({'seed': seed, 'elapsed_s': round(time.time() - t0, 2),
                            'repro_H32_maxabs': repro, 'trained32_vs_H32': first32,
                            'per_H': per_h})
        except Exception:
            results.append({'seed': seed, 'error': traceback.format_exc()[:1500]})
        print(json.dumps({k: results[-1].get(k) for k in ['seed', 'error', 'repro_H32_maxabs']}), flush=True)

    # verdict heuristic over valid seeds
    verdicts = {}
    for H in HORIZONS:
        if H == 32:
            verdicts['32'] = 'trained baseline'
            continue
        ratios = [r['per_H'][str(H)].get('tail_over_trained_jerk_ratio') for r in results
                  if 'per_H' in r and str(H) in r['per_H'] and r['per_H'][str(H)].get('tail_over_trained_jerk_ratio')]
        nans = sum(r['per_H'][str(H)]['nan_count'] for r in results if 'per_H' in r)
        gmins = [r['per_H'][str(H)]['grip_min'] for r in results if 'per_H' in r]
        gmaxs = [r['per_H'][str(H)]['grip_max'] for r in results if 'per_H' in r]
        med = float(np.median(ratios)) if ratios else None
        if nans > 0:
            verdicts[str(H)] = 'UNUSABLE: NaN/Inf present'
        elif med is not None and med > 3.0:
            verdicts[str(H)] = 'LIKELY_UNUSABLE: tail jerk %.1fx trained; OOD (positions %d+ untrained)' % (med, 32)
        else:
            verdicts[str(H)] = 'PLAUSIBLE_BUT_OOD: tail jerk %.2fx; needs closed-loop check; NOT a capability claim' % (med if med else -1)
    out = {'probe': 'H0_ZEROTRAIN_H_PROBE', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'checkpoint': CKPT, 'checkpoint_sha256': None,
           'checkpoint_is_f0': '/f0_v3_baseline/' in CKPT,
           'horizons': HORIZONS, 'seeds': SEEDS, 'denoise': T.DENOISE, 't_vid': T.T_VID,
           'note': 'exploratory OOD inference; positions 32+ never trained; NOT a 48/64 capability claim',
           'results': results, 'verdict': verdicts}
    try:
        out['checkpoint_sha256'] = T.sha256_file(CKPT)
    except Exception:
        pass
    np.savez_compressed(NPZ, **chunks)
    out['chunks_npz'] = NPZ
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({'H0_DONE': True, 'verdict': verdicts, 'out': OUT}), flush=True)


if __name__ == '__main__':
    main()
