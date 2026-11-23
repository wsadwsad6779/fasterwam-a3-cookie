import json
import os
import sys
import time
import traceback

sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
sys.path.insert(0, '/workspace')

import numpy as np
import torch
from omegaconf import OmegaConf

from t7_data import A3BatchSource, AH, T_VID
from t7_force_film import ForceFiLM

CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
OUT = '/workspace/fwv1/p_items/T7_TRAIN_PROBE.json'
B = 8
HIST = 8


def main():
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'batch_size': B}
    try:
        src = A3BatchSource(split='train', batch_size=B, seed=0)
        res['train_episodes'] = len(src.episodes)
        batches = []
        tdata = []
        for _ in range(3):
            t0 = time.time()
            batches.append(src.next_batch())
            tdata.append(round(time.time() - t0, 2))
        res['data_seconds'] = tdata
        b = batches[0]
        res['shapes'] = {k: list(v.shape) for k, v in b.items() if torch.is_tensor(v)}
        res['action_close_frac'] = float((b['action'][:, :, 6] > 0).float().mean())
        res['force_mask_mean'] = float(b['force_history_mask'].float().mean())
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
        model.mot.load_state_dict(sd['mot'], strict=False)
        model.proprio_encoder.load_state_dict(sd['proprio_encoder'], strict=False)
        res['create_s'] = round(time.time() - t0, 1)
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.action_expert.parameters():
            p.requires_grad_(True)
        for p in model.proprio_encoder.parameters():
            p.requires_grad_(True)
        model.vae.eval()
        model.train()
        film = ForceFiLM(force_dim=HIST * 4, hidden_dim=1024).cuda()
        state = {'z': None}
        orig_post = model.action_expert.post_dit

        def patched(tokens, pre_state):
            gamma, beta = film(state['z'])
            return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

        model.action_expert.post_dit = patched
        trainable = [p for p in model.parameters() if p.requires_grad] + list(film.parameters())
        res['trainable_params'] = int(sum(p.numel() for p in trainable))
        opt = torch.optim.AdamW(trainable, lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        steps = []
        for step in range(2):
            bb = batches[step]
            state['z'] = bb['force_history'].reshape(B, -1).cuda()
            sample = {'video': bb['video'].cuda(), 'context': bb['context'].cuda(),
                      'context_mask': bb['context_mask'].cuda(), 'action': bb['action'].cuda(),
                      'proprio': bb['proprio'].cuda(),
                      'action_is_pad': bb['action_is_pad'].cuda(),
                      'image_is_pad': bb['image_is_pad'].cuda()}
            t0 = time.time()
            opt.zero_grad()
            loss, ld = model.training_loss(sample)
            t_fwd = round(time.time() - t0, 2)
            t0 = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            t_step = round(time.time() - t0, 2)
            steps.append({'step': step + 1, 'loss': float(loss.detach()), 'dict': ld,
                          'fwd_s': t_fwd, 'bwd_step_s': t_step})
            print(json.dumps(steps[-1]), flush=True)
        res['steps'] = steps
        res['vram_peak_gb'] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
        res['est_21k_hours_compute'] = round(sum(s['fwd_s'] + s['bwd_step_s'] for s in steps) / 2 * 21000 / 3600 + 21000 * float(np.mean(tdata)) / 3600, 2)
        res['ok'] = all(np.isfinite(s['loss']) for s in steps)
    except Exception:
        res['error'] = traceback.format_exc()[:2500]
        res['ok'] = False
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({'ok': res.get('ok'), 'data_seconds': res.get('data_seconds'),
                      'steps': res.get('steps'), 'vram': res.get('vram_peak_gb'),
                      'trainable_M': (res.get('trainable_params') or 0) / 1e6,
                      'est_21k_hours': res.get('est_21k_hours_compute')}))


if __name__ == '__main__':
    main()