"""Bare-model MuJoCo evaluation: checkpoint -> 10 fixed seeds -> dual-view MP4 + strict metrics."""
import json
import os
import sys
import time
import traceback

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace/fwv1/sim_repo/src')
sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
sys.path.insert(0, '/workspace')

import numpy as np
import torch

from t7_force_film import ForceFiLM

CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
STATS = '/workspace/fwv1/data/a3_fw_845/dataset_stats.json'
FSTATS = '/workspace/fwv1/p_items/FINGERTIP_TRAIN_NORM_STATS.json'
CACHE_DIR = '/workspace/fwv1/data/text_embeds_cache'
SEEDS_MANIFEST = '/workspace/fwv1/p_items/EVAL_SEED_MANIFEST.json'
ROOT = '/workspace/fwv1/eval_fingertip_v1'
PROMPT = "A video recorded from a robot's point of view executing the following instruction: transfer 10 cookies into target box"
CONTEXT_LEN = 128
HIST = 8
T_VID = 9
MAX_STEPS = int(os.environ.get('T7_MAX_STEPS', '600'))
DENOISE = int(os.environ.get('T7_DENOISE', '5'))
ACTION_HORIZON = 32
TRANS_SCALE = 0.025
ROT_SCALE = 0.10
DT = 0.05


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def image_to_model(obs):
    import torch.nn.functional as Fn
    f = torch.tensor(obs['observation.images.front']).permute(2, 0, 1).float() / 255.0
    w = torch.tensor(obs['observation.images.left_wrist']).permute(2, 0, 1).float() / 255.0
    f = Fn.interpolate(f[None], size=(224, 224), mode='bilinear', align_corners=False)[0]
    w = Fn.interpolate(w[None], size=(224, 224), mode='bilinear', align_corners=False)[0]
    img = torch.cat([f, w], dim=2)
    return ((img - 0.5) / 0.5)[None]


class Stats:
    def __init__(self):
        st = json.load(open(STATS))
        a, s = st['action']['default'], st['state']['default']
        self.amin = np.asarray(a['global_min'], np.float32)
        self.amax = np.asarray(a['global_max'], np.float32)
        self.smin = np.asarray(s['global_min'], np.float32)
        self.smax = np.asarray(s['global_max'], np.float32)
        fs = json.load(open(FSTATS))
        self.fmean = np.asarray(fs['mean'], np.float32)
        self.fstd = np.maximum(np.asarray(fs['std'], np.float32), 1e-3)
        self.factive = (np.asarray(fs['std']) > 1e-6).astype(np.float32)

    @staticmethod
    def forward(x, lo, hi):
        rng = hi - lo
        rng = np.where(rng < 1e-4, 2.0, rng)
        scale = 2.0 / rng
        off = -1.0 - scale * lo
        return np.clip(x * scale + off, -5.0, 5.0), scale, off

    @staticmethod
    def backward(x, scale, off):
        return (x - off) / scale


def build_model(ckpt_path):
    from omegaconf import OmegaConf
    d = OmegaConf.load(CFG + '/data/a3_2cam.yaml')
    m = OmegaConf.load(CFG + '/model/fasterwam.yaml')
    cfg = OmegaConf.create({'data': d, 'model': m})
    try:
        OmegaConf.resolve(cfg)
    except Exception:
        pass
    import fasterwam.runtime as R
    model = R.create_fasterwam(
        model_id=WAN, tokenizer_model_id=WAN, tokenizer_max_len=128,
        video_dit_config=cfg.model.video_dit_config, proprio_dim=8,
        action_dit_config=cfg.model.action_dit_config,
        video_scheduler=cfg.model.video_scheduler,
        action_scheduler=cfg.model.action_scheduler,
        loss=cfg.model.loss, condition_layers=cfg.model.condition_layers,
        video_kv_fusion=cfg.model.video_kv_fusion,
        video_kv_fusion_init=cfg.model.video_kv_fusion_init,
        mot_checkpoint_mixed_attn=False, skip_dit_load_from_pretrain=True,
        load_text_encoder=False, redirect_common_files=False)
    sd = torch.load(BASE, map_location='cpu', mmap=True, weights_only=False)
    model.mot.load_state_dict(sd['mot'], strict=False)
    model.proprio_encoder.load_state_dict(sd['proprio_encoder'], strict=False)
    film = None
    state = {'z': None}
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if 'action_expert' in ck:
            model.action_expert.load_state_dict(ck['action_expert'])
        if 'proprio' in ck:
            model.proprio_encoder.load_state_dict(ck['proprio'])
        if 'film' in ck:
            film = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
            film.load_state_dict(ck['film'])
            film.eval()
            orig_post = model.action_expert.post_dit

            def patched(tokens, pre_state):
                if state['z'] is None:
                    return orig_post(tokens, pre_state)
                gamma, beta = film(state['z'])
                return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

            model.action_expert.post_dit = patched
    model.eval()
    if film is not None:
        film.eval()
    return model, film, state


def rollout(model, film, state, seed, stats, ctx, ctx_mask, out_dir, step_label):
    from a3_dual_arm_sim import A3CookieTransferEnv
    import imageio
    env = A3CookieTransferEnv(action_mode='cartesian_delta', render_mode='rgb_array', render_cameras=True)
    u = getattr(env, 'unwrapped', env)
    obs, _ = env.reset(seed=seed)
    os.makedirs(out_dir, exist_ok=True)
    frames_f, frames_w = [], []
    ring = np.zeros((HIST, 4), np.float32)
    rows = []
    info = {}
    close_cmd = 0
    grasps = 0
    prev_source = None
    hold = 0
    best_hold = 0
    exact_fill = False
    t = 0
    t0 = time.time()
    while t < MAX_STEPS:
        img = image_to_model(obs)
        prop_raw = np.asarray(obs['observation.state'], np.float64)[:8]
        prop, _, _ = stats.forward(prop_raw, stats.smin, stats.smax)
        state['z'] = torch.tensor(np.clip((ring - stats.fmean) / stats.fstd * stats.factive, -5, 5)
                                  .reshape(1, -1), dtype=torch.float32).cuda()
        with torch.no_grad():
            out = model.infer_action(prompt=None, context=ctx, context_mask=ctx_mask,
                                     input_image=img.cuda(), action_horizon=ACTION_HORIZON,
                                     num_video_frames=T_VID,
                                     proprio=torch.tensor(prop, dtype=torch.float32)[None].cuda(),
                                     num_inference_steps=DENOISE,
                                     seed=int(seed) * 100000 + t)
        act_norm = np.asarray(out['action'].float().cpu().numpy())
        if act_norm.ndim == 3:
            act_norm = act_norm[0]
        if act_norm.ndim != 2:
            raise ValueError('unexpected action shape: %s' % (act_norm.shape,))
        _, ascale, aoff = stats.forward(np.zeros(7, np.float32), stats.amin, stats.amax)
        act = stats.backward(act_norm, ascale, aoff)
        for k in range(ACTION_HORIZON):
            if t >= MAX_STEPS:
                break
            a = act[k]
            trans = np.clip(a[:3] * DT / TRANS_SCALE, -1, 1)
            rot = np.clip(a[3:6] * DT / ROT_SCALE, -1, 1)
            grip_cmd = -1.0 if a[6] >= 0 else 1.0
            cmd = np.r_[trans, rot, grip_cmd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            obs, r, term, trunc, info = env.step(cmd)
            pad = np.asarray(obs['observation.force'], np.float64)[12:16]
            ring = np.roll(ring, -1, axis=0)
            ring[-1] = pad
            frames_f.append(np.ascontiguousarray(obs['observation.images.front']))
            frames_w.append(np.ascontiguousarray(obs['observation.images.left_wrist']))
            src = info.get('cookies_in_source') if isinstance(info, dict) else None
            if prev_source is not None and src is not None and src < prev_source:
                grasps += prev_source - src
            prev_source = src
            rows.append({'t': t, 'grip_cmd': float(grip_cmd), 'opening': float(u._gripper_state('L')[0]),
                         'force': [float(x) for x in pad], 'in_target': info.get('cookies_in_target'),
                         'in_source': src, 'safety': info.get('safety_reason')})
            if info.get('cookies_in_target') == 10 and src == 0:
                hold += 1
                best_hold = max(best_hold, hold)
                if hold >= 20:
                    exact_fill = True
            else:
                hold = 0
            t += 1
            if term or trunc:
                break
        if term or trunc:
            break
    vf = os.path.join(out_dir, 'front.mp4')
    vw = os.path.join(out_dir, 'left_wrist.mp4')
    imageio.mimsave(vf, frames_f, fps=20, codec='libx264', quality=8)
    imageio.mimsave(vw, frames_w, fps=20, codec='libx264', quality=8)
    try:
        env.close()
    except Exception:
        pass
    pad_max = np.max(np.abs([r['force'] for r in rows]), axis=0).tolist() if rows else [0, 0, 0, 0]
    res = {'episode': int(seed), 'steps': t, 'exact_fill': bool(exact_fill),
           'best_hold': int(best_hold), 'grasps': int(grasps),
           'final_in_target': info.get('cookies_in_target'), 'final_in_source': info.get('cookies_in_source'),
           'close_cmd_frac': float(np.mean([r['grip_cmd'] < 0 for r in rows])) if rows else None,
           'pad_max': pad_max,
           'safety_reason': info.get('safety_reason'), 'terminated': bool(term), 'elapsed_s': round(time.time() - t0, 1),
           'videos': {'front': vf, 'left_wrist': vw},
           'events_json': os.path.join(out_dir, 'events.json')}
    json.dump({'step_label': step_label, 'seed': int(seed), 'rows': rows}, open(res['events_json'], 'w'))
    return res


def main():
    ckpt = os.environ.get('T7_CKPT', '')
    step_label = os.environ.get('T7_STEP', 'base')
    seeds_env = os.environ.get('T7_SEEDS', '')
    if seeds_env:
        seeds = [int(x) for x in seeds_env.split(',') if x.strip()]
    else:
        man = json.load(open(SEEDS_MANIFEST))
        seeds = man['milestone_seeds'].get(str(step_label), [7100]) if step_label != 'base' else [7001]
    out_root = os.path.join(ROOT, 'step_%06d' % (int(step_label) if step_label.isdigit() else 0))
    stats = Stats()
    cache = torch.load(os.path.join(CACHE_DIR, '%s.t5_len%d.wan22ti2v5b.pt' % (
        __import__('hashlib').sha256(PROMPT.encode()).hexdigest(), CONTEXT_LEN)), map_location='cpu')
    ctx = cache['context'][None].cuda()
    ctx_mask = cache['mask'].bool()[None].cuda()
    model, film, state = build_model(ckpt)
    results = []
    for seed in seeds:
        out_dir = os.path.join(out_root, 'seed_%05d' % seed)
        try:
            r = rollout(model, film, state, seed, stats, ctx, ctx_mask, out_dir, step_label)
        except Exception:
            r = {'episode': int(seed), 'error': traceback.format_exc()[:1500], 'exact_fill': False}
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ['episode', 'steps', 'exact_fill', 'final_in_target',
                                                'grasps', 'safety_reason', 'error']}), flush=True)
    summary = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'step_label': step_label, 'checkpoint': ckpt, 'checkpoint_sha256': sha256_file(ckpt) if ckpt and os.path.exists(ckpt) else None,
               'seeds': seeds, 'results': results,
               'exact_fill_count': sum(1 for r in results if r.get('exact_fill')),
               'exact_fill_rate': sum(1 for r in results if r.get('exact_fill')) / max(1, len(results)),
               'avg_in_target': float(np.mean([r.get('final_in_target') or 0 for r in results])),
               'safety_stop_count': sum(1 for r in results if r.get('safety_reason') is not None),
               'protocol': {'max_steps': MAX_STEPS, 'denoise_steps': DENOISE, 'action_horizon': ACTION_HORIZON,
                            'cameras': ['front', 'left_wrist']},
               'video_manifest': os.path.join(out_root, 'VIDEO_MANIFEST_step_%s.json' % step_label)}
    os.makedirs(out_root, exist_ok=True)
    json.dump(summary, open(os.path.join(out_root, 'SUMMARY_step_%s.json' % step_label), 'w'), indent=1)
    vids = []
    for r in results:
        if 'videos' in r:
            vids.append({'seed': r['episode'], 'front': r['videos']['front'],
                         'left_wrist': r['videos']['left_wrist'],
                         'frames': r['steps'], 'exact_fill': r['exact_fill'],
                         'in_target': r.get('final_in_target')})
    json.dump({'step_label': step_label, 'videos': vids, 'root': out_root},
              open(summary['video_manifest'], 'w'), indent=1)
    print(json.dumps({'step': step_label, 'exact_fill_rate': summary['exact_fill_rate'],
                      'avg_in_target': summary['avg_in_target']}))


if __name__ == '__main__':
    main()