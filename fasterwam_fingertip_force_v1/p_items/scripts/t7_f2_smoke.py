import os, sys, json, time, traceback

sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
sys.path.insert(0, '/workspace')

import numpy as np
import pyarrow.parquet as pq
import torch

from omegaconf import OmegaConf

CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
STATS = '/workspace/fwv1/p_items/FINGERTIP_TRAIN_NORM_STATS.json'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
OUT = '/workspace/fwv1/p_items/F2_FUSION_FWD_BWD_TEST.json'
RESUME_OUT = '/workspace/fwv1/p_items/F2_RESUME_TEST.json'
SAVE_DIR = '/workspace/fwv1/train_runs/t7_f2_smoke'
EPISODE = 0
T = 5
AH = 8
HIST = 8

from t7_force_film import ForceFiLM


def load_real_batch():
    z = np.load(LABELS)
    L = z['label']
    lab_start = z['ep_start_label'].astype(np.int64)
    ls = int(lab_start[EPISODE])
    action = torch.tensor(L[ls:ls + AH], dtype=torch.float32).unsqueeze(0)
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.state', 'observation.force'])
    E = np.asarray(t['episode_index'])
    S = np.asarray(t['observation.state'].to_pylist(), np.float64)
    fc = t['observation.force'].combine_chunks()
    F = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    order = np.argsort(E, kind='stable')
    E = E[order]
    S = S[order]
    F = F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    ep_ids = E[st]
    s = int(st[list(ep_ids).index(EPISODE)])
    proprio = torch.tensor(S[s:s + T], dtype=torch.float32).unsqueeze(0)
    stats = json.load(open(STATS))
    mean = np.asarray(stats['mean'])
    std = np.maximum(np.asarray(stats['std']), 1e-3)
    active = (np.asarray(stats['std']) > 1e-6).astype(np.float64)
    raw = F[s:s + HIST, 12:16]
    norm = np.clip((raw - mean) / std * active, -5, 5)
    force_hist = torch.tensor(norm, dtype=torch.float32).unsqueeze(0)
    return action, proprio, force_hist, S[s:s + T], F[s:s + HIST, 12:16]


def main():
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'variant': 'F2 zero-init Force-gated FiLM on action tokens',
           'batch': {'episode': EPISODE, 'action': 'real 7D labels (A3_845_7D_LABELS_V2)',
                     'proprio': 'real observation.state (8D)', 'force': 'real ch12-15 history (8 steps)',
                     'video': 'synthetic (1,3,5,64,64); context synthetic (1,16,4096)',
                     'honesty_note': 'video/context synthetic for smoke only; actions/proprio/force are real episode data'},
           'official_base': BASE}
    try:
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
            mot_checkpoint_mixed_attn=False,
            skip_dit_load_from_pretrain=True, load_text_encoder=False,
            redirect_common_files=False)
        sd = torch.load(BASE, map_location='cpu', mmap=True, weights_only=False)
        miss, unexp = model.mot.load_state_dict(sd['mot'], strict=False)
        pe_miss, pe_unexp = model.proprio_encoder.load_state_dict(sd['proprio_encoder'], strict=False)
        res['load'] = {'create_s': round(time.time() - t0, 1), 'mot_missing': len(miss),
                       'mot_unexpected': len(unexp), 'proprio_missing': len(pe_miss),
                       'proprio_unexpected': len(pe_unexp)}
        for p in model.parameters():
            p.requires_grad_(False)
        model.vae.eval()
        model.eval()
        action, proprio, force_hist, proprio_np, force_raw = load_real_batch()
        action = action.cuda(); proprio = proprio.cuda(); force_hist = force_hist.cuda()
        B = 1
        g = torch.Generator(device='cpu')
        g.manual_seed(0)
        sample = {'video': torch.rand(B, 3, T, 64, 64, generator=g).cuda(),
                  'context': torch.randn(B, 16, 4096, generator=g).cuda(),
                  'context_mask': torch.ones(B, 16, dtype=torch.bool).cuda(),
                  'action': action, 'proprio': proprio,
                  'action_is_pad': torch.zeros(B, AH, dtype=torch.bool).cuda(),
                  'image_is_pad': torch.zeros(B, T, dtype=torch.bool).cuda()}
        model.train()

        def seeded():
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)

        seeded()
        base_loss, base_dict = model.training_loss(sample)
        base_loss_v = float(base_loss.detach())
        film = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
        for p in film.parameters():
            p.requires_grad_(True)
        film.train()
        z = force_hist.reshape(B, -1)
        state = {'z': z}
        orig_post = model.action_expert.post_dit

        def patched(tokens, pre_state):
            gamma, beta = film(state['z'])
            tokens = ForceFiLM.modulate(tokens, gamma, beta)
            return orig_post(tokens, pre_state)

        model.action_expert.post_dit = patched
        seeded()
        film_loss, film_dict = model.training_loss(sample)
        film_loss_v = float(film_loss.detach())
        res['identity'] = {'base_loss': base_loss_v, 'film_loss_zero_init': film_loss_v,
                           'abs_diff': abs(base_loss_v - film_loss_v),
                           'identity_ok': abs(base_loss_v - film_loss_v) < 1e-9,
                           'base_dict': base_dict, 'film_dict': film_dict}
        params = [p for p in film.parameters()]
        res['film_params'] = int(sum(p.numel() for p in params))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        film_loss.backward()
        res['bwd_s'] = round(time.time() - t0, 2)
        gn = {}
        for name, p in film.named_parameters():
            if p.grad is not None:
                gn[name] = float(p.grad.detach().float().norm())
        res['film_grad_norms'] = gn
        res['force_encoder_grad_nonzero'] = bool(gn.get('net.0.weight', 0.0) > 0)
        res['film_head_grad_nonzero'] = bool(gn.get('net.2.weight', 0.0) > 0)
        opt = torch.optim.AdamW(params, lr=1e-2)
        opt.step()
        with torch.no_grad():
            gamma_after, beta_after = film(z)
        res['after_step'] = {'gamma_absmax': float(gamma_after.abs().max()),
                             'beta_absmax': float(beta_after.abs().max()),
                             'nonzero': bool(gamma_after.abs().max() > 0 and beta_after.abs().max() > 0)}
        res['step1_note'] = ('zero-init FiLM output layer blocks gradient to the force encoder at step 1 '
                             '(dL/dW_encoder = W_out^T * ... = 0); this is expected, encoder learns from step 2')
        seeded()
        film_loss2, _ = model.training_loss(sample)
        film.zero_grad()
        film_loss2.backward()
        gn2 = {}
        for name, p in film.named_parameters():
            if p.grad is not None:
                gn2[name] = float(p.grad.detach().float().norm())
        res['step2_grad_norms'] = gn2
        res['force_encoder_grad_nonzero'] = bool(gn2.get('net.0.weight', 0.0) > 0)
        opt.step()
        with torch.no_grad():
            gamma_after, beta_after = film(z)
        res['after_step2'] = {'gamma_absmax': float(gamma_after.abs().max()),
                              'beta_absmax': float(beta_after.abs().max())}
        seeded()
        with torch.no_grad():
            loss2, dict2 = model.training_loss(sample)
        res['post_step_loss'] = float(loss2.detach())
        res['post_step_loss_changed'] = bool(abs(float(loss2.detach()) - film_loss_v) > 0)
        res['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        os.makedirs(SAVE_DIR, exist_ok=True)
        ckpt = os.path.join(SAVE_DIR, 'f2_film_state.pt')
        payload = {'film': film.state_dict(), 'optimizer': opt.state_dict(),
                   'rng_torch': torch.get_rng_state(),
                   'rng_cuda': torch.cuda.get_rng_state_all(),
                   'force_stats': STATS, 'episode': EPISODE,
                   'step': 2, 'base_loss': base_loss_v}
        torch.save(payload, ckpt)
        import hashlib
        h = hashlib.sha256()
        with open(ckpt, 'rb') as f:
            for c in iter(lambda: f.read(1 << 22), b''):
                h.update(c)
        ckpt_sha = h.hexdigest()
        res['checkpoint'] = {'path': ckpt, 'bytes': os.path.getsize(ckpt), 'sha256': ckpt_sha}
        film2 = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
        film2.load_state_dict(torch.load(ckpt, map_location='cuda', weights_only=False)['film'])
        with torch.no_grad():
            g2, b2 = film2(z)
        rres = {'worker': 'task7', 'ts': res['ts'], 'checkpoint': res['checkpoint'],
                'gamma_max_diff': float((g2 - gamma_after).abs().max()),
                'beta_max_diff': float((b2 - beta_after).abs().max())}
        rres['pass'] = bool(rres['gamma_max_diff'] == 0 and rres['beta_max_diff'] == 0)
        json.dump(rres, open(RESUME_OUT, 'w'), indent=1)
        res['resume'] = rres
        res['checks'] = {
            'identity_at_init': res['identity']['identity_ok'],
            'force_encoder_grad_nonzero_step2': res['force_encoder_grad_nonzero'],
            'film_head_grad_nonzero': res['film_head_grad_nonzero'],
            'film_nonzero_after_step': res['after_step']['nonzero'],
            'loss_changed_after_step': res['post_step_loss_changed'],
            'resume_exact': rres['pass'],
        }
        res['verdict'] = 'pass' if all(res['checks'].values()) else 'fail'
    except Exception:
        res['error'] = traceback.format_exc()[:2500]
        res['verdict'] = 'error'
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': res['verdict'], 'checks': res.get('checks'),
                      'identity': res.get('identity', {}).get('abs_diff'),
                      'loss': res.get('post_step_loss'), 'vram': res.get('vram_peak_gb')}))


if __name__ == '__main__':
    main()