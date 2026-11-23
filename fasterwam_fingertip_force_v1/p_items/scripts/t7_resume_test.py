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

from t7_data import A3BatchSource
from t7_force_film import ForceFiLM

CFG = '/workspace/fwv1/repo/stage_40k/fw/configs'
WAN = '/workspace/fwv1/wan_base'
BASE = '/workspace/fwv1/hf_public/eval_restore_20261117/base_step_021700.pt'
OUT = '/workspace/fwv1/p_items/RESUME_FUNCTIONAL_TEST.json'
CKPT = '/workspace/fwv1/train_runs/t7_resume_test/state.pt'
B = 8
HIST = 8
N_STEPS = 6
SAVE_AT = 3
LR = 1e-4


def build(seed=0):
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
    state = {'z': None}
    orig_post = model.action_expert.post_dit

    def patched(tokens, pre_state):
        gamma, beta = film(state['z'])
        return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

    model.action_expert.post_dit = patched
    src = A3BatchSource(split='train', batch_size=B, seed=seed)
    return model, film, state, src


def trainable_state(model, film):
    st = {k: v.detach().cpu() for k, v in model.action_expert.state_dict().items()}
    st.update({'proprio.' + k: v.detach().cpu() for k, v in model.proprio_encoder.state_dict().items()})
    st.update({'film.' + k: v.detach().cpu() for k, v in film.state_dict().items()})
    return st


def load_trainable(model, film, st):
    ae = {k: st[k] for k in model.action_expert.state_dict().keys()}
    model.action_expert.load_state_dict(ae)
    pe = {k.split('.', 1)[1]: st[k] for k in st if k.startswith('proprio.')}
    model.proprio_encoder.load_state_dict(pe)
    fm = {k.split('.', 1)[1]: st[k] for k in st if k.startswith('film.')}
    film.load_state_dict(fm)


def make_step(model, film, state, opt, sched, src, sample):
    state['z'] = sample['force_history'].reshape(B, -1).cuda()
    s = {'video': sample['video'].cuda(), 'context': sample['context'].cuda(),
         'context_mask': sample['context_mask'].cuda(), 'action': sample['action'].cuda(),
         'proprio': sample['proprio'].cuda(),
         'action_is_pad': sample['action_is_pad'].cuda(),
         'image_is_pad': sample['image_is_pad'].cuda()}
    opt.zero_grad()
    loss, ld = model.training_loss(s)
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad] + list(film.parameters()), 1.0)
    opt.step()
    sched.step()
    return float(loss.detach())


def run(mode='full'):
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'batch_size': B, 'steps': N_STEPS, 'save_at': SAVE_AT, 'mode': mode}
    model, film, state, src = build(seed=0)
    params = [p for p in model.parameters() if p.requires_grad] + list(film.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    if mode == 'restore':
        payload = torch.load(CKPT, map_location='cpu', weights_only=False)
        load_trainable(model, film, payload['trainable'])
        opt.load_state_dict(payload['optimizer'])
        sched.load_state_dict(payload['scheduler'])
        torch.set_rng_state(payload['rng_torch'])
        torch.cuda.set_rng_state_all(payload['rng_cuda'])
        for _ in range(SAVE_AT):
            src.next_batch()
        start_step = SAVE_AT + 1
    else:
        start_step = 1
    losses = []
    for step in range(start_step, N_STEPS + 1):
        sample = src.next_batch()
        loss = make_step(model, film, state, opt, sched, src, sample)
        lr = sched.get_last_lr()[0]
        losses.append({'step': step, 'loss': loss, 'lr': lr})
        if mode == 'full' and step == SAVE_AT:
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            torch.save({'trainable': trainable_state(model, film),
                        'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                        'rng_torch': torch.get_rng_state(),
                        'rng_cuda': torch.cuda.get_rng_state_all(),
                        'step': step}, CKPT)
    res['losses'] = losses
    res['checkpoint'] = {'path': CKPT, 'bytes': os.path.getsize(CKPT) if os.path.exists(CKPT) else None}
    return res


def main():
    out = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        full = run('full')
        out['continuous'] = full
        import hashlib
        h = hashlib.sha256()
        with open(CKPT, 'rb') as f:
            for c in iter(lambda: f.read(1 << 22), b''):
                h.update(c)
        out['checkpoint_sha256'] = h.hexdigest()
        restored = run('restore')
        out['restored'] = restored
        cont = {x['step']: x['loss'] for x in full['losses'] if x['step'] > SAVE_AT}
        rest = {x['step']: x['loss'] for x in restored['losses']}
        diffs = {str(k): abs(cont[k] - rest[k]) for k in cont}
        out['loss_abs_diff'] = diffs
        out['max_abs_diff'] = max(diffs.values()) if diffs else None
        out['checks'] = {'resume_ran': len(rest) == N_STEPS - SAVE_AT,
                         'loss_match_lt_1e6': out['max_abs_diff'] is not None and out['max_abs_diff'] < 1e-6,
                         'optimizer_scheduler_rng_saved': True}
        out['verdict'] = 'pass' if all(out['checks'].values()) else 'fail'
    except Exception:
        out['error'] = traceback.format_exc()[:2500]
        out['verdict'] = 'error'
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps({'verdict': out['verdict'], 'max_abs_diff': out.get('max_abs_diff'),
                      'continuous': out.get('continuous', {}).get('losses'),
                      'restored': out.get('restored', {}).get('losses')}))


if __name__ == '__main__':
    main()