"""Segmented 21k training driver: one segment per invocation (e.g. 0->3000).

Full resumable checkpoint = trainable weights (action expert + proprio + F2 FiLM)
+ optimizer + scheduler + RNG + manifest. Frozen base weights are referenced by sha256.
"""
import hashlib
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
BASE_SHA = '01be4efca565d1890fd8a7dc1507b01c00c8cc63c16ae252287492ba9ac7550d'
RUN_DIR = os.environ.get('T7_RUN_DIR', '/workspace/fwv1/train_runs/t7_21k')
LOG = os.environ.get('T7_LOG', '/workspace/fwv1/logs/t7_train.log')
METRICS = RUN_DIR + '/metrics.jsonl'
STATE = os.environ.get('T7_STATE', '/workspace/fwv1/p_items/POD_JOB_STATE.json')
HIST = 8
B = int(os.environ.get('T7_BATCH', '8'))
LR = float(os.environ.get('T7_LR', '1e-4'))
TOTAL = 21000
START = int(os.environ.get('T7_START_STEP', '0'))
END = int(os.environ.get('T7_END_STEP', '3000'))
CKPT_EVERY = int(os.environ.get('T7_CKPT_EVERY', '500'))
AUX_W = float(os.environ.get('T7_AUX_FORCE_W', '0.05'))


def log(m, fh=None):
    print(m, flush=True)
    with open(LOG, 'a') as f:
        f.write(m + '\n')
    if fh is not None:
        fh.write(m + '\n')
        fh.flush()


def st(status, step, note=''):
    d = {'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'pid': os.getpid(),
         'stage': 'train', 'log_path': LOG, 'status': status, 'step': step,
         'expected_artifacts': [RUN_DIR], 'next': 'milestone eval at 3k boundary', 'note': note}
    json.dump(d, open(STATE, 'w'), indent=1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def build():
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
    force_head = torch.nn.Sequential(
        torch.nn.LayerNorm(1024),
        torch.nn.Linear(1024, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 1),
    ).cuda()
    torch.nn.init.zeros_(force_head[-1].weight)
    torch.nn.init.zeros_(force_head[-1].bias)
    state = {'z': None, 'feat': None, 'B': None}
    orig_post = model.action_expert.post_dit

    def patched(tokens, pre_state):
        if tokens.dim() == 3:
            dim = 1 if (state['B'] is None or tokens.shape[0] == state['B']) else 0
            state['feat'] = tokens.mean(dim=dim)
        else:
            state['feat'] = tokens.mean(dim=0)
        gamma, beta = film(state['z'])
        return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

    model.action_expert.post_dit = patched
    return model, film, force_head, state


def save(path, model, film, force_head, opt, sched, step, losses):
    payload = {'action_expert': model.action_expert.state_dict(),
               'proprio': model.proprio_encoder.state_dict(),
               'film': film.state_dict(),
               'force_head': force_head.state_dict(),
               'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
               'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all(),
               'step': step, 'base_sha256': BASE_SHA,
               'losses': losses[-50:]}
    tmp = path + '.tmp'
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return os.path.getsize(path)


def main():
    os.makedirs(RUN_DIR, exist_ok=True)
    st('running', START, note='segment %d->%d' % (START, END))
    log('START segment %d->%d batch=%d lr=%g' % (START, END, B, LR))
    model, film, force_head, state = build()
    params = ([p for p in model.parameters() if p.requires_grad]
              + list(film.parameters()) + list(force_head.parameters()))
    opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
    resume = os.environ.get('T7_RESUME', '')
    start_step = int(os.environ.get('T7_START_STEP', '0'))
    losses = []
    ck = None
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location='cpu', weights_only=False)
        model.action_expert.load_state_dict(ck['action_expert'])
        model.proprio_encoder.load_state_dict(ck['proprio'])
        film.load_state_dict(ck['film'])
        if 'force_head' in ck:
            force_head.load_state_dict(ck['force_head'])
        else:
            log('WARN checkpoint has no force_head; aux head starts from init')
        start_step = int(ck['step'])
        losses = ck.get('losses', [])
        log('RESUMED from %s at step %d' % (resume, start_step))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (start_step + s + 1) / 500.0) *
        (0.5 * (1 + np.cos(np.pi * min(1.0, (start_step + s) / TOTAL)))))
    if ck is not None:
        try:
            opt.load_state_dict(ck['optimizer'])
        except Exception as exc:
            log('WARN optimizer state not loaded: %s' % exc)
        sched.load_state_dict(ck['scheduler'])
        torch.set_rng_state(ck['rng_torch'])
        torch.cuda.set_rng_state_all(ck['rng_cuda'])
    src = A3BatchSource(split='train', batch_size=B, seed=start_step, balanced=True)
    step = start_step
    while step < END:
        bb = src.next_batch()
        state['z'] = bb['force_history'].reshape(B, -1).cuda()
        state['B'] = B
        state['feat'] = None
        sample = {'video': bb['video'].cuda(), 'context': bb['context'].cuda(),
                  'context_mask': bb['context_mask'].cuda(), 'action': bb['action'].cuda(),
                  'proprio': bb['proprio'].cuda(),
                  'action_is_pad': bb['action_is_pad'].cuda(),
                  'image_is_pad': bb['image_is_pad'].cuda()}
        t0 = time.time()
        opt.zero_grad()
        loss_main, ld = model.training_loss(sample)
        if state['feat'] is not None and 'force_target' in bb:
            pred_f = force_head(state['feat'].float()).squeeze(-1)
            loss_force = torch.nn.functional.mse_loss(pred_f, bb['force_target'].cuda())
            loss = loss_main + AUX_W * loss_force
        else:
            loss_force = None
            loss = loss_main
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        rec = {'step': step, 'loss': float(loss.detach()), 'loss_main': float(loss_main.detach()),
               'loss_force': None if loss_force is None else float(loss_force.detach()),
               'loss_video': ld.get('loss_video'),
               'loss_action': ld.get('loss_action'), 'grad_norm': float(gn),
               'lr': sched.get_last_lr()[0], 'sec': round(time.time() - t0, 2),
               'episode': bb['meta']['episode'],
               'close_frac': float((bb['grip_raw'] <= 0.6).float().mean()),
               'open_frac': float((bb['grip_raw'] >= 0.8).float().mean()),
               'release_frac': float(((bb['grip_raw'] > 0.35) & (bb['grip_raw'] < 0.7)).float().mean()),
               'force_drop_frac': float(bb['force_dropped'].float().mean()),
               'feat_dim': int(state['feat'].shape[-1]) if state['feat'] is not None else None,
               'film_grad': None}
        if step % 10 == 0 or step == END:
            with open(METRICS, 'a') as f:
                f.write(json.dumps(rec) + '\n')
            log(json.dumps(rec, default=str))
            st('running', step)
        if step % CKPT_EVERY == 0 or step == END:
            p = os.path.join(RUN_DIR, 'latest.pt')
            sz = save(p, model, film, force_head, opt, sched, step, losses + [rec['loss']])
            log('CKPT %s step=%d bytes=%d' % (p, step, sz))
        if not np.isfinite(rec['loss']):
            st('failed', step, note='non-finite loss')
            log('FAIL non-finite loss at step %d' % step)
            return
    p = os.path.join(RUN_DIR, 'step_%06d.pt' % END)
    sz = save(p, model, film, force_head, opt, sched, END, losses)
    log('SEGMENT_DONE step=%d ckpt=%s bytes=%d' % (END, p, sz))
    st('done', END, note='segment complete')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('TOPFAIL')
        log(traceback.format_exc()[:2500])
        st('failed', -1, note=traceback.format_exc()[-500:])