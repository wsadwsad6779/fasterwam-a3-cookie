#!/usr/bin/env python3
"""B5: TRUE gradient conflict between L_grip (action dim6) and L_force (aux force head).
F2 step_015000 DOES contain force_head weights (LayerNorm1024->Linear256->SiLU->Linear1).
We replicate the trainer's forward: state['feat']=tokens.mean(over token dim) is the head input,
L_force = MSE(force_head(feat), force_target). One forward/backward, NO optimizer step.
cos(g_grip, g_force) on the shared action_expert trunk answers 'do the grip and force objectives conflict?'"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

sys.path.insert(0, '/workspace')

CKPT = '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt'
OUT = os.environ.get('B5_OUT', '/workspace/fwv1/p_items/B5_LGRIP_LFORCE_GRADCOS.json')
B = 2
SEED = 20261125


def flat(params):
    # Over the FULL param set; missing .grad -> zeros, so every loss yields the SAME length vector.
    if not params:
        return None
    dev = params[0].device
    parts = []
    for p in params:
        if p.grad is not None:
            parts.append(p.grad.detach().reshape(-1))
        else:
            parts.append(torch.zeros(p.numel(), device=dev, dtype=p.dtype))
    return torch.cat(parts)


def cos(a, b):
    if a is None or b is None:
        return None
    na = float(a.norm()); nb = float(b.norm())
    if na < 1e-12 or nb < 1e-12:
        return None
    return float(torch.dot(a, b) / (na * nb))


def main():
    from t7_data import A3BatchSource
    spec = importlib.util.spec_from_file_location('t7ev', '/workspace/_t7_eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    model, film, state = ev.build_model(CKPT)
    # force head exactly as the trainer defines it (match ckpt keys 0/1/3)
    force_head = nn.Sequential(
        nn.LayerNorm(1024), nn.Linear(1024, 256), nn.SiLU(), nn.Linear(256, 1),
    ).cuda()
    ck = torch.load(CKPT, map_location='cpu', weights_only=False)
    lforce_status = 'COMPUTED'
    lforce_reason = None
    missing = []
    try:
        if 'force_head' not in ck:
            raise KeyError('force_head')
        force_head.load_state_dict(ck['force_head'])
    except Exception as exc:
        lforce_status = 'NOT_VERIFIED'
        lforce_reason = 'force_head load failed: %s' % exc
        missing = list(ck.get('force_head', {}).keys())
    # wrap post_dit so state['feat'] is captured (mean over token dim), then let the eval film patch run
    ev_patched = model.action_expert.post_dit

    def my_patch(tokens, pre_state):
        if tokens.dim() == 3:
            dim = 1 if (state.get('B') is None or tokens.shape[0] == state['B']) else 0
            state['feat'] = tokens.mean(dim=dim)
        else:
            state['feat'] = tokens.mean(dim=0)
        return ev_patched(tokens, pre_state)

    model.action_expert.post_dit = my_patch
    model.train()
    src = A3BatchSource(split='val', batch_size=B, seed=SEED, balanced=False)
    bb = src.next_batch()
    state['z'] = bb['force_history'].reshape(B, -1).float().cuda()
    state['B'] = B
    state['feat'] = None
    sample = {'video': bb['video'].cuda(), 'context': bb['context'].cuda(),
              'context_mask': bb['context_mask'].cuda(), 'action': bb['action'].cuda(),
              'proprio': bb['proprio'].cuda(), 'action_is_pad': bb['action_is_pad'].cuda(),
              'image_is_pad': bb['image_is_pad'].cuda()}
    cap = []
    orig = Fn.mse_loss

    def patched(pred, target, *a, **k):
        if torch.is_tensor(pred) and pred.dim() == 3 and pred.shape[-1] == 7:
            cap.append((pred, target))
        return orig(pred, target, *a, **k)

    Fn.mse_loss = patched
    model.zero_grad()
    loss_main, ld = model.training_loss(sample)
    Fn.mse_loss = orig
    if not cap:
        raise SystemExit('NO_ACTION_LOSS_CAPTURED')
    pred, target = cap[-1]
    ae_params = list(model.action_expert.parameters())
    has_ft = ('force_target' in bb) and (state['feat'] is not None) and (lforce_status == 'COMPUTED')
    loss_force = None
    if has_ft:
        pred_f = force_head(state['feat'].float()).squeeze(-1)
        tgt_f = bb['force_target'].float().cuda()
        loss_force = Fn.mse_loss(pred_f, tgt_f)
    elif lforce_status == 'COMPUTED':
        lforce_status = 'NOT_VERIFIED'
        lforce_reason = 'force_target missing in val batch or feat not captured'
        missing = ([] if 'force_target' in bb else ['batch.force_target']) + ([] if state['feat'] is not None else ['state.feat'])
    # grads
    model.zero_grad(); force_head.zero_grad()
    ((pred[..., 6] - target[..., 6]) ** 2).mean().backward(retain_graph=True)
    g6 = flat(ae_params)
    model.zero_grad(); force_head.zero_grad()
    ((pred[..., :6] - target[..., :6]) ** 2).mean().backward(retain_graph=True)
    g_other = flat(ae_params)
    g_force_ae = None; g_force_fh = None; cos_g6_force = None
    if loss_force is not None:
        model.zero_grad(); force_head.zero_grad()
        loss_force.backward(retain_graph=True)
        g_force_ae = flat(ae_params)
        g_force_fh = flat(list(force_head.parameters()))
        cos_g6_force = cos(g6, g_force_ae)
    norms = {'g_grip': (float(g6.norm()) if g6 is not None else None),
             'g_other6': (float(g_other.norm()) if g_other is not None else None),
             'g_force_ae': (float(g_force_ae.norm()) if g_force_ae is not None else None),
             'g_force_head': (float(g_force_fh.norm()) if g_force_fh is not None else None)}
    od = {'artifact': 'B5_LGRIP_LFORCE_GRADCOS',
          'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'checkpoint': CKPT, 'batch': {'split': 'val', 'seed': SEED},
          'L_force_status': lforce_status,
          'L_force_reason': lforce_reason,
          'L_force_missing_keys': missing,
          'loss_force_value': (float(loss_force.detach()) if loss_force is not None else None),
          'cos_grip_vs_force_ae': cos_g6_force,
          'cos_grip_vs_other6': cos(g6, g_other),
          'grad_norms': norms,
          'ratio_g_grip_to_g_other6': (norms['g_grip'] / norms['g_other6']) if (norms['g_grip'] and norms['g_other6']) else None,
          'force_head_loaded_from_ckpt': (lforce_status != 'NOT_VERIFIED' or 'force_head' in ck),
          'note': 'TRUE L_force reconstructed from the trainer forward (feat=tokens.mean(token-dim)); COS is AUX-weight invariant; read-only, no optimizer step',
          'read_only': True}
    json.dump(od, open(OUT, 'w'), indent=1)
    print(json.dumps({'B5_DONE': True, 'L_force_status': lforce_status,
                      'cos_grip_vs_force_ae': cos_g6_force,
                      'cos_grip_vs_other6': od['cos_grip_vs_other6']}), flush=True)


if __name__ == '__main__':
    main()
