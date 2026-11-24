"""Bare-model MuJoCo evaluation: checkpoint -> 10 fixed seeds -> dual-view MP4 + strict metrics."""
import hashlib
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
STATS = '/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'
FSTATS = '/workspace/fwv1/p_items/FINGERTIP_FORCE_NORMALIZATION_V2.json'
CACHE_DIR = '/workspace/fwv1/data/text_embeds_cache'
SEEDS_MANIFEST = '/workspace/fwv1/p_items/EVAL_SEED_MANIFEST.json'
SCENE_MANIFEST = '/workspace/fwv1/p_items/EVAL_SCENE_MANIFEST_V3_1.json'
SIM_CONFIG = os.environ.get('T7_SIM_CONFIG', '/workspace/fwv1/sim_repo/configs/cookie_batch.yaml')
ROOT = '/workspace/fwv1/eval_fingertip_v1'
PROMPT = "A video recorded from a robot's point of view executing the following instruction: transfer 10 cookies into target box"
CONTEXT_LEN = 128
HIST = 8
T_VID = 9
MAX_STEPS = int(os.environ.get('T7_MAX_STEPS', '600'))
DENOISE = int(os.environ.get('T7_DENOISE', '5'))
FORCE_ZERO = os.environ.get('T7_FORCE_ZERO', '0') == '1'
SKIP_FILM = os.environ.get('T7_SKIP_FILM', '0') == '1'
ACTION_HORIZON = 32
TRANS_SCALE = 0.025
ROT_SCALE = 0.10
LABEL_TRANS_SCALE = 0.05
LABEL_ROT_SCALE = 0.5
REPLAN = 10
DT = 0.05


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def scene_hash(rm):
    import hashlib
    cp = np.asarray(rm.get('cookie_positions', []), np.float64)
    sb = np.asarray(rm.get('source_bin_position', []), np.float64)
    tb = np.asarray(rm.get('target_bin_position', []), np.float64)
    h = hashlib.sha1()
    h.update(np.round(cp, 6).astype(np.float32).tobytes())
    h.update(np.round(sb, 6).astype(np.float32).tobytes())
    h.update(np.round(tb, 6).astype(np.float32).tobytes())
    h.update(json.dumps(rm.get('cameras'), sort_keys=True, default=str).encode())
    h.update(json.dumps(rm.get('light_diffuse') or rm.get('light'), sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _panel_font(size=12):
    from PIL import ImageFont
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def force_panel(hist, step, seed, in_target, in_source, opening, grip_cmd, safety,
                w=320, h=256):
    from PIL import Image, ImageDraw
    img = Image.new('RGB', (w, h), (14, 14, 14))
    d = ImageDraw.Draw(img)
    f = _panel_font(12)
    fs = _panel_font(11)
    x0, y0, x1, y1 = 42, 20, w - 8, h - 50
    ymax = 3.0
    d.text((6, 3), 'grip force (N)  step %d  seed %d' % (step, seed), fill=(230, 230, 230), font=f)
    for v in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        y = int(y1 - (v / ymax) * (y1 - y0))
        d.line([(x0, y), (x1, y)], fill=(55, 55, 55))
        d.text((6, y - 6), '%.1f' % v, fill=(170, 170, 170), font=fs)
    yth = int(y1 - (0.5 / ymax) * (y1 - y0))
    for x in range(x0, x1, 8):
        d.line([(x, yth), (min(x + 4, x1), yth)], fill=(220, 70, 70))
    N = 120
    data = hist[-N:] if hist else []
    if data:
        for ci, col in ((0, (80, 200, 255)), (1, (255, 170, 60))):
            pts = []
            for i, pair in enumerate(data):
                v = min(float(pair[ci]), ymax)
                x = x0 + (i / max(1, N - 1)) * (x1 - x0)
                y = y1 - (v / ymax) * (y1 - y0)
                pts.append((x, y))
            if len(pts) > 1:
                d.line(pts, fill=col, width=2)
        cur = data[-1]
        peak = max(max(p) for p in data)
    else:
        cur = (0.0, 0.0)
        peak = 0.0
    d.text((x0, y1 + 3), 'L inner %.2f  L outer %.2f N' % (cur[0], cur[1]), fill=(80, 200, 255), font=f)
    d.text((x0, y1 + 17), 'peak %.2f N (y 0-3N) | 0.5N carry line' % peak, fill=(220, 180, 120), font=fs)
    d.text((x0, y1 + 31), ('open %.2f  gcmd %+.2f  inT %s  src %s' % (
        opening, grip_cmd, in_target, in_source))[:52], fill=(200, 200, 200), font=fs)
    if safety:
        d.text((x0, 6), ('SAFETY: %s' % safety)[:46], fill=(255, 80, 80), font=fs)
    d.rectangle([x0 - 1, y0 - 1, x1 + 1, y1 + 1], outline=(90, 90, 90))
    return np.asarray(img)


def composite_frame(obs, hist, step, seed, info, opening, grip_cmd):
    wimg = np.ascontiguousarray(obs['observation.images.left_wrist'])
    fimg = np.ascontiguousarray(obs['observation.images.front'])
    panel = force_panel(hist, step, seed, info.get('cookies_in_target'),
                        info.get('cookies_in_source'), opening, grip_cmd,
                        info.get('safety_reason'))
    div = np.full((wimg.shape[0], 2, 3), 255, np.uint8)
    return np.hstack([wimg, div, fimg, div, panel])


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
        self.fstd = np.maximum(np.asarray(fs['std'], np.float32), float(fs.get('std_floor', 0.05)))
        self.factive = np.asarray(fs.get('active', (np.asarray(fs['std']) > 1e-6)), np.float32)

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


def rollout(model, film, state, seed, stats, ctx, ctx_mask, out_dir, step_label,
            scene_manifest_path, scene_entry):
    from a3_dual_arm_sim import A3CookieTransferEnv
    import imageio
    env = A3CookieTransferEnv(config=SIM_CONFIG, action_mode='cartesian_delta',
                              render_mode='rgb_array', render_cameras=True)
    u = getattr(env, 'unwrapped', env)
    obs, _ = env.reset(seed=seed, options=dict(scene_entry['options']))
    rm = dict(getattr(u, 'randomization_metadata', {}) or {})
    scene_actual = scene_hash(rm)
    scene_ok = bool(scene_actual == scene_entry['hash'])
    if not scene_ok:
        raise RuntimeError('scene hash mismatch seed=%d expected=%s actual=%s'
                           % (seed, scene_entry['hash'], scene_actual))
    os.makedirs(out_dir, exist_ok=True)
    frames_c = []
    force_hist = []
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
        zfeat = np.clip((ring - stats.fmean) / stats.fstd * stats.factive, -5, 5).reshape(1, -1)
        if FORCE_ZERO:
            zfeat = np.zeros_like(zfeat)
        state['z'] = None if SKIP_FILM else torch.tensor(zfeat, dtype=torch.float32).cuda()
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
        for k in range(min(ACTION_HORIZON, REPLAN)):
            if t >= MAX_STEPS:
                break
            a = act[k]
            trans = np.clip(a[:3] * (LABEL_TRANS_SCALE / TRANS_SCALE), -1, 1)
            rot = np.clip(a[3:6] * (LABEL_ROT_SCALE / ROT_SCALE), -1, 1)
            grip_cmd = float(np.clip(2.0 * a[6] - 1.0, -1.0, 1.0))
            cmd = np.r_[trans, rot, grip_cmd, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            obs, r, term, trunc, info = env.step(cmd)
            pad = np.asarray(obs['observation.force'], np.float64)[12:16]
            ring = np.roll(ring, -1, axis=0)
            ring[-1] = pad
            force_hist.append((float(pad[0]), float(pad[1])))
            frames_c.append(composite_frame(obs, force_hist, t, seed, info if isinstance(info, dict) else {},
                                            float(u._gripper_state('L')[0]), grip_cmd))
            src = info.get('cookies_in_source') if isinstance(info, dict) else None
            if prev_source is not None and src is not None and src < prev_source:
                grasps += prev_source - src
            prev_source = src
            rows.append({'t': t, 'grip_cmd': float(grip_cmd), 'opening': float(u._gripper_state('L')[0]),
                         'force': [float(x) for x in pad], 'in_target': info.get('cookies_in_target'),
                         'in_source': src, 'safety': info.get('safety_reason'),
                         'act_dp_max': float(np.abs(a[:3]).max()), 'act_rot_max': float(np.abs(a[3:6]).max()),
                         'act_grip_raw': float(a[6]), 'opening_cmd': float(a[6]),
                         'eef_after': [float(x) for x in np.asarray(obs['observation.eef_pose']).reshape(-1)[:3]]})
            if info.get('cookies_in_target') == 10 and src == 70:
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
    vc = os.path.join(out_dir, 'composite.mp4')
    imageio.mimsave(vc, frames_c, fps=20, codec='libx264', quality=8)
    try:
        env.close()
    except Exception:
        pass
    pad_max = np.max(np.abs([r['force'] for r in rows]), axis=0).tolist() if rows else [0, 0, 0, 0]
    import math as _m
    dp_mean = float(np.mean([r['act_dp_max'] for r in rows])) if rows else None
    dp_max = float(np.max([r['act_dp_max'] for r in rows])) if rows else None
    rot_mean = float(np.mean([r['act_rot_max'] for r in rows])) if rows else None
    grip_close = float(np.mean([r.get('opening_cmd', 1.0) <= 0.6 for r in rows])) if rows else None
    ocmd = np.asarray([r.get('opening_cmd', 1.0) for r in rows]) if rows else np.zeros(0)
    open_cmd_frac = float((ocmd >= 0.8).mean()) if len(ocmd) else None
    reopen_events = int((np.diff((ocmd >= 0.8).astype(int)) == 1).sum()) if len(ocmd) > 1 else 0
    path = 0.0
    for i in range(1, len(rows)):
        a0 = np.asarray(rows[i-1]['eef_after']); a1 = np.asarray(rows[i]['eef_after'])
        path += float(np.linalg.norm(a1 - a0))
    pads = np.asarray([max(r['force']) for r in rows], np.float64) if rows else np.zeros(0)
    cmd_close = np.asarray([r['grip_cmd'] < 0 for r in rows], bool) if rows else np.zeros(0, bool)
    rel_mask = (ocmd > 0.35) & (ocmd < 0.7)
    force_stats = {
        'contact_frac': float((pads > 0.2).mean()) if len(pads) else None,
        'strong_contact_frac': float((pads > 0.5).mean()) if len(pads) else None,
        'contact_force_median': float(np.median(pads[pads > 0.2])) if (pads > 0.2).any() else None,
        'close_cmd_force_median': float(np.median(pads[cmd_close])) if cmd_close.any() else None,
        'carry_force_frac_gt_0p5': float((pads[cmd_close] > 0.5).mean()) if cmd_close.any() else None,
        'carry_force_frac_gt_0p2': float((pads[cmd_close] > 0.2).mean()) if cmd_close.any() else None,
        'release_cmd_force_median': float(np.median(pads[rel_mask])) if rel_mask.any() else None,
        'force_max': float(pads.max()) if len(pads) else None,
    }
    res = {'episode': int(seed), 'steps': t, 'exact_fill': bool(exact_fill),
           'action_stats': {'dp_absmean': dp_mean, 'dp_absmax': dp_max, 'rot_absmean': rot_mean,
                            'grip_close_frac': grip_close, 'open_cmd_frac': open_cmd_frac,
                            'reopen_events': reopen_events, 'eef_path_m': round(path, 4)},
           'force_stats': force_stats,
           'scene': {'seed': int(seed), 'profile': scene_entry.get('profile'),
                     'expected_hash': scene_entry['hash'], 'actual_hash': scene_actual,
                     'match': scene_ok, 'manifest': scene_manifest_path},
           'best_hold': int(best_hold), 'grasps': int(grasps),
           'final_in_target': info.get('cookies_in_target'), 'final_in_source': info.get('cookies_in_source'),
           'close_cmd_frac': float(np.mean([r['grip_cmd'] < 0 for r in rows])) if rows else None,
           'pad_max': pad_max,
           'safety_reason': info.get('safety_reason'), 'terminated': bool(term), 'elapsed_s': round(time.time() - t0, 1),
           'videos': {'composite': vc},
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
        seeds = man['milestone_seeds'].get(str(step_label), [7100]) if step_label != 'base' else [7100]
    sm_path = os.environ.get('T7_SCENE_MANIFEST', SCENE_MANIFEST)
    sm = json.load(open(sm_path))
    canonical = hashlib.sha256(json.dumps(
        {'base_options': sm['base_options'],
         'scenes': [{'seed': s['seed'], 'hash': s['hash']} for s in sm['scenes']]},
        sort_keys=True).encode()).hexdigest()
    if sm.get('sim_config_sha256') and sm['sim_config_sha256'] != sha256_file(SIM_CONFIG):
        raise RuntimeError('eval sim config does not match the frozen manifest')
    if canonical != sm.get('freeze_sha256'):
        raise RuntimeError('scene manifest freeze mismatch: %s != %s'
                           % (canonical, sm.get('freeze_sha256')))
    scene_map = {int(s['seed']): s for s in sm['scenes']}
    if len({s['hash'] for s in sm['scenes']}) != len(sm['scenes']):
        raise RuntimeError('duplicate scene hashes in frozen manifest')
    missing = [s for s in seeds if s not in scene_map]
    if missing:
        raise RuntimeError('eval seeds missing from frozen scene manifest: %s' % missing)
    sub = os.environ.get('T7_OUT_SUBDIR', 'step_%06d' % (int(step_label) if step_label.isdigit() else 0))
    out_root = os.path.join(ROOT, sub)
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
            r = rollout(model, film, state, seed, stats, ctx, ctx_mask, out_dir, step_label,
                        sm_path, scene_map[seed])
        except Exception:
            r = {'episode': int(seed), 'error': traceback.format_exc()[:1500], 'exact_fill': False}
        results.append(r)
        print(json.dumps({k: r.get(k) for k in ['episode', 'steps', 'exact_fill', 'final_in_target',
                                                'grasps', 'safety_reason', 'error']}), flush=True)
    summary = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'step_label': step_label, 'checkpoint': ckpt, 'checkpoint_sha256': sha256_file(ckpt) if ckpt and os.path.exists(ckpt) else None,
               'seeds': seeds, 'results': results,
               'action_stats_mean': {
                   'dp_absmean': float(np.mean([r['action_stats']['dp_absmean'] for r in results if r.get('action_stats')])) if any(r.get('action_stats') for r in results) else None,
                   'grip_close_frac': float(np.mean([r['action_stats']['grip_close_frac'] for r in results if r.get('action_stats')])) if any(r.get('action_stats') for r in results) else None,
                   'eef_path_m': float(np.mean([r['action_stats']['eef_path_m'] for r in results if r.get('action_stats')])) if any(r.get('action_stats') for r in results) else None,
                   'open_cmd_frac': float(np.mean([r['action_stats']['open_cmd_frac'] for r in results if r.get('action_stats')])) if any(r.get('action_stats') for r in results) else None,
                   'reopen_events_mean': float(np.mean([r['action_stats']['reopen_events'] for r in results if r.get('action_stats')])) if any(r.get('action_stats') for r in results) else None},
               'force_stats_mean': {
                   'contact_frac': float(np.mean([r['force_stats']['contact_frac'] for r in results if r.get('force_stats') and r['force_stats']['contact_frac'] is not None])) if any(r.get('force_stats') and r['force_stats']['contact_frac'] is not None for r in results) else None,
                   'strong_contact_frac': float(np.mean([r['force_stats']['strong_contact_frac'] for r in results if r.get('force_stats') and r['force_stats']['strong_contact_frac'] is not None])) if any(r.get('force_stats') and r['force_stats']['strong_contact_frac'] is not None for r in results) else None,
                   'carry_force_frac_gt_0p5': float(np.mean([r['force_stats']['carry_force_frac_gt_0p5'] for r in results if r.get('force_stats') and r['force_stats']['carry_force_frac_gt_0p5'] is not None])) if any(r.get('force_stats') and r['force_stats']['carry_force_frac_gt_0p5'] is not None for r in results) else None},
               'exact_fill_count': sum(1 for r in results if r.get('exact_fill')),
               'exact_fill_rate': sum(1 for r in results if r.get('exact_fill')) / max(1, len(results)),
               'avg_in_target': float(np.mean([r.get('final_in_target') or 0 for r in results])),
               'safety_stop_count': sum(1 for r in results if r.get('safety_reason') is not None),
               'protocol': {'max_steps': MAX_STEPS, 'denoise_steps': DENOISE, 'action_horizon': ACTION_HORIZON, 'replan': REPLAN,
                            'cameras': ['front', 'left_wrist']},
               'video_manifest': os.path.join(out_root, 'VIDEO_MANIFEST_step_%s.json' % step_label),
               'scene_manifest': {
                   'path': sm_path, 'file_sha256': sha256_file(sm_path),
                   'freeze_sha256': sm.get('freeze_sha256'),
                   'scenes': len(sm['scenes']),
                   'distinct_hashes': len({s['hash'] for s in sm['scenes']}),
                   'checks': [{'seed': r['episode'],
                               'expected': (r.get('scene') or {}).get('expected_hash'),
                               'actual': (r.get('scene') or {}).get('actual_hash'),
                               'match': (r.get('scene') or {}).get('match')} for r in results],
                   'all_match': all((r.get('scene') or {}).get('match') for r in results)},
               'force_zero': FORCE_ZERO}
    os.makedirs(out_root, exist_ok=True)
    json.dump(summary, open(os.path.join(out_root, 'SUMMARY_step_%s.json' % step_label), 'w'), indent=1)
    vids = []
    for r in results:
        if 'videos' in r:
            vids.append({'seed': r['episode'], 'composite': r['videos']['composite'],
                         'frames': r['steps'], 'exact_fill': r['exact_fill'],
                         'in_target': r.get('final_in_target'),
                         'panel': 'left_wrist|front|force(ch12/ch13, 0-3N, 0.5N line)'})
    json.dump({'step_label': step_label, 'videos': vids, 'root': out_root, 'force_zero': FORCE_ZERO},
              open(summary['video_manifest'], 'w'), indent=1)
    print(json.dumps({'step': step_label, 'exact_fill_rate': summary['exact_fill_rate'],
                      'avg_in_target': summary['avg_in_target'],
                      'scene_checks_pass': summary['scene_manifest']['all_match']}))


if __name__ == '__main__':
    main()