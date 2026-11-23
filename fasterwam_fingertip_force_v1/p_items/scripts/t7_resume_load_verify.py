"""Verify the live latest.pt can be strictly restored and used for one fwd/bwd without writing."""
import hashlib
import json
import os
import sys
import time
import traceback

sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
sys.path.insert(0, '/workspace')

import torch

from t7_data import A3BatchSource
from t7_force_film import ForceFiLM

CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
CKPT = '/workspace/fwv1/train_runs/t7_21k/latest.pt'
OUT = '/workspace/fwv1/p_items/RESUME_LOAD_VERIFY.json'
B = 2
HIST = 8


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
    from omegaconf import OmegaConf
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'ckpt': CKPT}
    try:
        res['ckpt_sha256'] = sha256_file(CKPT)
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
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.action_expert.parameters():
            p.requires_grad_(True)
        for p in model.proprio_encoder.parameters():
            p.requires_grad_(True)
        model.vae.eval()
        model.train()
        film = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
        ck = torch.load(CKPT, map_location='cpu', weights_only=False)
        model.action_expert.load_state_dict(ck['action_expert'], strict=True)
        model.proprio_encoder.load_state_dict(ck['proprio'], strict=True)
        film.load_state_dict(ck['film'], strict=True)
        res['strict_loads'] = {'action_expert': True, 'proprio': True, 'film': True}
        res['ckpt_step'] = int(ck['step'])
        params = [p for p in model.parameters() if p.requires_grad] + list(film.parameters())
        opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
        start = int(ck['step'])
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (start + s + 1) / 500.0) *
            (0.5 * (1 + __import__('math').cos(__import__('math').pi * min(1.0, (start + s) / 21000)))))
        opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_torch'])
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
        res['optimizer_scheduler_rng_loaded'] = True
        film_state = {'z': None}
        orig_post = model.action_expert.post_dit

        def patched(tokens, pre_state):
            gamma, beta = film(film_state['z'])
            return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

        model.action_expert.post_dit = patched
        src = A3BatchSource(split='train', batch_size=B, seed=start)
        bb = src.next_batch()
        film_state['z'] = bb['force_history'].reshape(B, -1).cuda()
        sample = {'video': bb['video'].cuda(), 'context': bb['context'].cuda(),
                  'context_mask': bb['context_mask'].cuda(), 'action': bb['action'].cuda(),
                  'proprio': bb['proprio'].cuda(),
                  'action_is_pad': bb['action_is_pad'].cuda(),
                  'image_is_pad': bb['image_is_pad'].cuda()}
        opt.zero_grad()
        loss, ld = model.training_loss(sample)
        loss.backward()
        gn = {n: float(p.grad.norm()) for n, p in film.named_parameters() if p.grad is not None}
        res['forward_loss'] = float(loss.detach())
        res['loss_dict'] = ld
        res['film_grads'] = gn
        res['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res['checks'] = {'strict_action_expert': True, 'strict_proprio': True, 'strict_film': True,
                         'loss_finite': bool(loss.detach() == loss.detach()),
                         'grads_present': all(v > 0 for v in gn.values())}
        res['verdict'] = 'pass' if all(res['checks'].values()) else 'fail'
    except Exception:
        res['error'] = traceback.format_exc()[:2000]
        res['verdict'] = 'error'
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': res['verdict'], 'step': res.get('ckpt_step'),
                      'loss': res.get('forward_loss'), 'vram': res.get('vram_peak_gb')}))


if __name__ == '__main__':
    main()