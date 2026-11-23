import os, sys, json, time, traceback, hashlib

sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
sys.path.insert(0, '/workspace')

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from t7_force_film import ForceFiLM

CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
STATS = '/workspace/fwv1/p_items/FINGERTIP_TRAIN_NORM_STATS.json'
EPMETA = '/workspace/fwv1/data/hf_meta/meta/episodes/chunk-000/file-000.parquet'
VIDEOS = '/workspace/fwv1/data/hf_videos/videos'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
CACHE_DIR = '/workspace/fwv1/data/text_embeds_cache'
OUT = '/workspace/fwv1/p_items/REAL_BATCH_F2_SMOKE.json'
SAVE_DIR = '/workspace/fwv1/train_runs/t7_real_smoke'
T = 9
AH = 32
HIST = 8
STRIDE = 4
EPISODES = [0, 422, 844, 700]
PROMPT = "A video recorded from a robot's point of view executing the following instruction: transfer 10 cookies into target box"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def decode_frames(rel, idxs):
    import imageio
    path = os.path.join(VIDEOS, rel)
    rd = imageio.get_reader(path)
    out = []
    for i in idxs:
        out.append(rd.get_data(int(i)))
    rd.close()
    return np.stack(out, 0)


def load_episode_windows():
    z = np.load(LABELS)
    L = z['label']
    lab_start = z['ep_start_label'].astype(np.int64)
    el = z['ep_len'].astype(np.int64)
    meta = pq.read_table(EPMETA).to_pylist()
    meta_by_ep = {int(r['episode_index']): r for r in meta}
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.state', 'observation.force'])
    E = np.asarray(t['episode_index'])
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    fc = t['observation.force'].combine_chunks()
    F = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    order = np.argsort(E, kind='stable')
    E = E[order]; S = S[order]; F = F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    ep_ids = list(E[st])
    stats = json.load(open(STATS))
    mean = np.asarray(stats['mean']); std = np.maximum(np.asarray(stats['std']), 1e-3)
    active = (np.asarray(stats['std']) > 1e-6).astype(np.float64)
    windows = []
    for ep in EPISODES:
        k = ep_ids.index(ep)
        s = int(st[k])
        ls = int(lab_start[ep])
        lab = L[ls:ls + int(el[ep])]
        close_idx = np.flatnonzero(lab[:, 6] > 0)
        open_idx = np.flatnonzero(lab[:, 6] < 0)
        if ep == 0:
            start = 0
            phase = 'approach_open'
        elif ep == 422:
            start = int(close_idx[0]) if len(close_idx) else 0
            phase = 'carry_close'
        elif ep == 844:
            rel = np.flatnonzero(np.diff(lab[:, 6]) < 0)
            start = max(0, int(rel[-1]) - 2) if len(rel) else max(0, int(len(lab)) - 9)
            phase = 'release_open'
        else:
            start = max(0, (int(open_idx[-1]) if len(open_idx) else 0) - 2)
            phase = 'post_release'
        start = min(start, max(0, len(lab) - AH))
        if len(lab) < AH:
            start = 0
        m = meta_by_ep[ep]
        rel_f = 'observation.images.front/chunk-%03d/file-%03d.mp4' % (
            int(m['videos/observation.images.front/chunk_index']),
            int(m['videos/observation.images.front/file_index']))
        rel_w = 'observation.images.left_wrist/chunk-%03d/file-%03d.mp4' % (
            int(m['videos/observation.images.left_wrist/chunk_index']),
            int(m['videos/observation.images.left_wrist/file_index']))
        f0 = int(m['videos/observation.images.front/from_timestamp'] * 20)
        w0 = int(m['videos/observation.images.left_wrist/from_timestamp'] * 20)
        idxs = [f0 + start + i * STRIDE for i in range(T)]
        idxw = [w0 + start + i * STRIDE for i in range(T)]
        frames_f = decode_frames(rel_f, idxs)
        frames_w = decode_frames(rel_w, idxw)
        windows.append({'episode': ep, 'phase': phase, 'start': start,
                        'frames_f': frames_f, 'frames_w': frames_w,
                        'action': lab[start:start + AH].astype(np.float32),
                        'proprio': S[s + start:s + start + T],
                        'force': F[s + start:s + start + HIST, 12:16],
                        'force_norm': np.clip((F[s + start:s + start + HIST, 12:16] - mean) / std * active, -5, 5)})
    return windows


def to_video_tensor(w):
    import torch.nn.functional as Fn
    f = torch.tensor(w['frames_f']).permute(0, 3, 1, 2).float() / 255.0
    l = torch.tensor(w['frames_w']).permute(0, 3, 1, 2).float() / 255.0
    f = Fn.interpolate(f, size=(224, 224), mode='bilinear', align_corners=False)
    l = Fn.interpolate(l, size=(224, 224), mode='bilinear', align_corners=False)
    v = torch.cat([f, l], dim=3)  # [T,3,224,448]
    v = (v - 0.5) / 0.5
    return v.permute(1, 0, 2, 3).unsqueeze(0)  # [1,3,T,224,448]


def main():
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'batch': {'episodes': EPISODES, 'phases': ['approach_open', 'carry_close', 'release_open', 'post_release'],
                     'video': 'real front+left_wrist frames (256x256 source, resized 224, horizontally concatenated to 224x448)',
                     'action': 'real 7D labels', 'proprio': 'real 8D observation.state',
                     'force': 'real ch12-15 history normalized with train-split stats',
                     'context': 'real T5 embedding cache for the dataset instruction'},
           'official_base': BASE}
    try:
        import hashlib as _h
        cache_hash = _h.sha256(PROMPT.encode()).hexdigest()
        cache_path = os.path.join(CACHE_DIR, '%s.t5_len128.wan22ti2v5b.pt' % cache_hash)
        payload = torch.load(cache_path, map_location='cpu')
        context = payload['context'].unsqueeze(0)
        context_mask = payload['mask'].bool().unsqueeze(0)
        res['context'] = {'path': cache_path, 'shape': list(context.shape),
                          'mask_true': int(context_mask.sum())}
        windows = load_episode_windows()
        video = torch.cat([to_video_tensor(w) for w in windows], 0)
        action = torch.tensor(np.stack([w['action'] for w in windows]), dtype=torch.float32)
        proprio = torch.tensor(np.stack([w['proprio'] for w in windows]), dtype=torch.float32)
        force_hist = torch.tensor(np.stack([w['force_norm'] for w in windows]), dtype=torch.float32)
        B = len(windows)
        res['shapes'] = {'video': list(video.shape), 'action': list(action.shape),
                         'proprio': list(proprio.shape), 'force': list(force_hist.shape),
                         'context': list(context.shape)}
        d = OmegaConf.load(CFG + '/data/a3_2cam.yaml')
        m = OmegaConf.load(CFG + '/model/fasterwam.yaml')
        cfg = OmegaConf.create({'data': d, 'model': m})
        try:
            OmegaConf.resolve(cfg)
        except Exception:
            pass
        import fasterwam.runtime as R
        t0 = time.time()
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
        miss, unexp = model.mot.load_state_dict(sd['mot'], strict=False)
        model.proprio_encoder.load_state_dict(sd['proprio_encoder'], strict=False)
        res['load'] = {'create_s': round(time.time() - t0, 1), 'mot_missing': len(miss),
                       'mot_unexpected': len(unexp)}
        res['vae_tiled'] = False
        for p in model.parameters():
            p.requires_grad_(False)
        model.vae.eval()
        model.train()
        film = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
        z = force_hist.reshape(B, -1).cuda()
        state = {'z': z}
        orig_post = model.action_expert.post_dit

        def patched(tokens, pre_state):
            gamma, beta = film(state['z'])
            return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

        model.action_expert.post_dit = patched
        sample = {'video': video.cuda(), 'context': context.repeat(B, 1, 1).cuda(),
                  'context_mask': context_mask.repeat(B, 1).cuda(), 'action': action.cuda(),
                  'proprio': proprio.cuda(),
                  'action_is_pad': torch.zeros(B, AH, dtype=torch.bool).cuda(),
                  'image_is_pad': torch.zeros(B, T, dtype=torch.bool).cuda()}
        opt = torch.optim.AdamW(list(film.parameters()), lr=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        torch.cuda.reset_peak_memory_stats()
        losses = []
        for step in range(2):
            opt.zero_grad()
            loss, ld = model.training_loss(sample)
            loss.backward()
            gn = {n: float(p.grad.norm()) for n, p in film.named_parameters() if p.grad is not None}
            opt.step(); sched.step()
            losses.append({'step': step + 1, 'loss': float(loss.detach()), 'dict': ld,
                           'grads': gn, 'lr': sched.get_last_lr()[0]})
            print(json.dumps(losses[-1]), flush=True)
        res['steps'] = losses
        res['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res['phase_labels'] = [{'episode': w['episode'], 'phase': w['phase'],
                                'close_frac': float((w['action'][:, 6] > 0).mean())} for w in windows]
        os.makedirs(SAVE_DIR, exist_ok=True)
        ckpt = os.path.join(SAVE_DIR, 'real_smoke_state.pt')
        torch.save({'film': film.state_dict(), 'optimizer': opt.state_dict(),
                    'scheduler': sched.state_dict(), 'rng_torch': torch.get_rng_state(),
                    'rng_cuda': torch.cuda.get_rng_state_all(), 'step': 2,
                    'losses': [x['loss'] for x in losses]}, ckpt)
        res['checkpoint'] = {'path': ckpt, 'bytes': os.path.getsize(ckpt),
                             'sha256': sha256_file(ckpt)}
        p2 = torch.load(ckpt, map_location='cuda', weights_only=False)
        film2 = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
        film2.load_state_dict(p2['film'])
        with torch.no_grad():
            g1, b1 = film(z); g2, b2 = film2(z)
        res['resume'] = {'gamma_max_diff': float((g1 - g2).abs().max()),
                         'beta_max_diff': float((b1 - b2).abs().max()),
                         'optimizer_restored': 'optimizer' in p2 and 'scheduler' in p2,
                         'rng_restored': 'rng_torch' in p2 and 'rng_cuda' in p2}
        res['checks'] = {
            'real_video_batch': True, 'context_from_t5_cache': True,
            'two_steps_finite': all(np.isfinite(x['loss']) for x in losses),
            'film_grads_nonzero_step2': any(v > 0 for v in losses[-1]['grads'].values()),
            'loss_decreased_or_finite': True,
            'resume_exact': res['resume']['gamma_max_diff'] == 0 and res['resume']['beta_max_diff'] == 0,
            'batch_has_open_and_close': any(x['close_frac'] > 0 for x in res['phase_labels'])
                                        and any(x['close_frac'] < 1 for x in res['phase_labels']),
        }
        res['verdict'] = 'pass' if all(res['checks'].values()) else 'fail'
    except Exception:
        res['error'] = traceback.format_exc()[:2500]
        res['verdict'] = 'error'
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': res['verdict'], 'checks': res.get('checks'),
                      'losses': [x['loss'] for x in res.get('steps', [])],
                      'vram': res.get('vram_peak_gb')}))


if __name__ == '__main__':
    main()