import os, sys, json, time, traceback

sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
OUT = '/workspace/fwv1/p_items/P4_FULL_SMOKE.json'
LOG = '/workspace/fwv1/logs/p4_full_t7.log'


def log(m):
    print(m, flush=True)
    try:
        with open(LOG, 'a') as f:
            f.write(m + '\n')
    except Exception:
        pass


import torch
from omegaconf import OmegaConf

res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
       'official_base': BASE, 'wan_dir': WAN}
try:
    d = OmegaConf.load(CFG + '/data/a3_2cam.yaml')
    m = OmegaConf.load(CFG + '/model/fasterwam.yaml')
    cfg = OmegaConf.create({'data': d, 'model': m})
    try:
        OmegaConf.resolve(cfg)
    except Exception as ex:
        res['omega_resolve_error'] = str(ex)[:200]
    import fasterwam.runtime as R
    t0 = time.time()
    model = R.create_fasterwam(
        model_id=WAN,
        tokenizer_model_id=WAN,
        tokenizer_max_len=128,
        video_dit_config=cfg.model.video_dit_config,
        proprio_dim=8,
        action_dit_config=cfg.model.action_dit_config,
        video_scheduler=cfg.model.video_scheduler,
        action_scheduler=cfg.model.action_scheduler,
        loss=cfg.model.loss,
        condition_layers=cfg.model.condition_layers,
        video_kv_fusion=cfg.model.video_kv_fusion,
        video_kv_fusion_init=cfg.model.video_kv_fusion_init,
        mot_checkpoint_mixed_attn=False,
        skip_dit_load_from_pretrain=True,
        load_text_encoder=False,
        redirect_common_files=False,
    )
    res['create_s'] = round(time.time() - t0, 1)
    res['n_params_M'] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
    res['device'] = str(getattr(model, 'device', '?'))
    res['proprio_dim'] = getattr(model, 'proprio_dim', None)
    res['action_dim'] = int(getattr(model.action_expert, 'action_dim', -1))
    res['model_paths'] = {k: str(v) for k, v in getattr(model, 'model_paths', {}).items()
                          if isinstance(v, (str, int, float))}
    force_keys = [k for k, _ in model.named_parameters() if 'force' in k.lower()]
    res['force_param_count'] = len(force_keys)
    sd = torch.load(BASE, map_location='cpu', mmap=True, weights_only=False)
    res['base_keys'] = list(sd.keys())
    res['base_mot_n'] = len(sd['mot'])
    res['base_step'] = sd.get('step')
    res['base_dtype'] = str(sd.get('torch_dtype'))
    miss, unexp = model.mot.load_state_dict(sd['mot'], strict=False)
    res['mot_missing'] = len(miss)
    res['mot_unexpected'] = len(unexp)
    res['mot_missing_sample'] = list(miss)[:8]
    res['mot_unexpected_sample'] = list(unexp)[:8]
    if hasattr(model, 'proprio_encoder') and model.proprio_encoder is not None and 'proprio_encoder' in sd:
        pe_miss, pe_unexp = model.proprio_encoder.load_state_dict(sd['proprio_encoder'], strict=False)
        res['proprio_missing'] = len(pe_miss)
        res['proprio_unexpected'] = len(pe_unexp)
    for p in model.vae.parameters():
        p.requires_grad_(False)
    res['vae_frozen'] = True
    model.train()
    model.vae.eval()
    B, T, H, Wd = 1, 5, 64, 64
    g = torch.Generator(device='cpu')
    g.manual_seed(0)
    sample = {
        'video': torch.rand(B, 3, T, H, Wd, generator=g),
        'context': torch.randn(B, 16, 4096, generator=g),
        'context_mask': torch.ones(B, 16, dtype=torch.bool),
        'action': torch.randn(B, 8, 7, generator=g),
        'proprio': torch.randn(B, T, 8, generator=g),
        'action_is_pad': torch.zeros(B, 8, dtype=torch.bool),
        'image_is_pad': torch.zeros(B, T, dtype=torch.bool),
    }
    res['input_shapes'] = {k: list(v.shape) for k, v in sample.items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    loss, ld = model.training_loss(sample)
    res['fwd_s'] = round(time.time() - t0, 2)
    res['loss_total'] = float(loss.detach())
    res['loss_dict'] = ld
    t0 = time.time()
    loss.backward()
    res['bwd_s'] = round(time.time() - t0, 2)
    tot = 0.0
    n = 0
    nonfinite = 0
    for p in model.parameters():
        if p.grad is not None:
            n += 1
            gv = p.grad.detach().float().pow(2).sum().item()
            if not (gv == gv and gv != float('inf')):
                nonfinite += 1
            tot += gv
    res['grad_params'] = n
    res['grad_norm'] = round(tot ** 0.5, 6)
    res['grad_nonfinite_params'] = nonfinite
    if torch.cuda.is_available():
        res['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    ok_loss = res['loss_total'] == res['loss_total']
    res['ok'] = bool(res.get('mot_missing') == 0 and res.get('mot_unexpected') == 0
                     and nonfinite == 0 and ok_loss)
except Exception:
    res['error'] = traceback.format_exc()[:2000]
    res['ok'] = False

hist = []
try:
    old = json.load(open(OUT))
    hist = old.get('history', [])
except Exception:
    old = {}
hist.append(res)
out = {'worker': 'task7', 'stage': 'P4_full_model_fwd_bwd_official_base',
       'attempts': len(hist), 'latest': res, 'history': hist[-5:]}
try:
    json.dump(out, open(OUT, 'w'), indent=1)
except Exception:
    log('WRITE_OUT_FAIL ' + traceback.format_exc()[:300])
log(json.dumps({'ok': res.get('ok'), 'loss_total': res.get('loss_total'),
                'fwd_s': res.get('fwd_s'), 'bwd_s': res.get('bwd_s'),
                'vram_peak_gb': res.get('vram_peak_gb'),
                'error': (res.get('error') or '')[:400]}))
print('DONE')