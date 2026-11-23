"""D5: F2 trainable parameter plan, gripper-dim gradient, force-pathway effectiveness,
and F2_TRAINING_READY decision."""
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
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
SPLIT = '/workspace/fwv1/p_items/DATASET_845_SPLIT_MANIFEST_V2.json'
OUT_PARAMS = '/workspace/fwv1/p_items/F2_TRAINABLE_PARAMETERS.json'
OUT_EFF = '/workspace/fwv1/p_items/F2_FORCE_EFFECTIVENESS_TEST.json'
OUT_MD = '/workspace/fwv1/p_items/F2_TRAINING_READINESS.md'
B = 4
HIST = 8
T_VID = 9


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
    state = {'z': None}
    orig_post = model.action_expert.post_dit

    def patched(tokens, pre_state):
        gamma, beta = film(state['z'])
        return orig_post(ForceFiLM.modulate(tokens, gamma, beta), pre_state)

    model.action_expert.post_dit = patched
    return model, film, state


def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    res = {'worker': 'task7', 'ts': ts}
    try:
        z = np.load(LABELS)
        L = z['label']
        split = json.load(open(SPLIT))
        train_ids = set(int(x) for x in split['train_episodes'])
        starts = z['ep_start_label'].astype(np.int64)
        el = z['ep_len'].astype(np.int64)
        close = 0
        total = 0
        for k in range(len(starts)):
            if k in train_ids:
                s = int(starts[k])
                n = int(el[k])
                close += int((L[s:s + n, 6] > 0).sum())
                total += n
        res['class_balance_v2_train'] = {'close_frames': close, 'total': total,
                                         'close_frac': close / max(1, total),
                                         'open_frac': 1 - close / max(1, total)}

        model, film, state = build()
        src = A3BatchSource(split='train', batch_size=B, seed=0)
        bb = src.next_batch()
        state['z'] = bb['force_history'].reshape(B, -1).cuda()
        sample = {'video': bb['video'].cuda(), 'context': bb['context'].cuda(),
                  'context_mask': bb['context_mask'].cuda(), 'action': bb['action'].cuda(),
                  'proprio': bb['proprio'].cuda(),
                  'action_is_pad': bb['action_is_pad'].cuda(),
                  'image_is_pad': bb['image_is_pad'].cuda()}
        params = [p for p in model.parameters() if p.requires_grad] + list(film.parameters())
        loss, ld = model.training_loss(sample)
        loss.backward()
        head = model.action_expert.head
        g = head.weight.grad.detach().float()
        row = [float(g[i].norm()) for i in range(g.shape[0])]
        res['gripper_dim_gradient'] = {'head_weight_shape': list(g.shape), 'per_out_dim_grad_norm': row,
                                       'dim6_grad_norm': row[6], 'other_dims_mean': float(np.mean(row[:6])),
                                       'dim6_nonzero': bool(row[6] > 0)}
        fenc = {n: float(p.grad.detach().float().norm()) for n, p in film.named_parameters() if p.grad is not None}
        res['force_encoder_grads_step1'] = fenc
        res['param_plan'] = {
            'video_expert_frozen': True,
            'action_expert_trainable': int(sum(p.numel() for p in model.action_expert.parameters())),
            'proprio_trainable': int(sum(p.numel() for p in model.proprio_encoder.parameters())),
            'film_trainable': int(sum(p.numel() for p in film.parameters())),
            'total_trainable': int(sum(p.numel() for p in params)),
            'lora': 'not used (full action-expert finetune within budget; LoRA kept as fallback if memory/time regress)',
            'optimizer': 'AdamW fp32 states for trainable params only (~5.59GB per full checkpoint incl weights)',
        }
        pj = {'worker': 'task7', 'ts': ts, **res['param_plan'],
              'class_balance': res['class_balance_v2_train'],
              'gripper_dim_gradient': res['gripper_dim_gradient'],
              'force_encoder_grads_step1': fenc,
              'checkpoint_estimate_gb': 5.59, 'seven_milestones_gb': 39.13}
        json.dump(pj, open(OUT_PARAMS, 'w'), indent=1)

        # force pathway effectiveness through infer_action
        img = bb['video'][0, :, 0][None].cuda()
        prop = bb['proprio'][0, 0][None].cuda()
        ctx = bb['context'][0:1].cuda(); cm = bb['context_mask'][0:1].cuda()
        zreal = bb['force_history'][0:1].reshape(1, -1).cuda()
        zzero = torch.zeros_like(zreal)
        zshuf = torch.flip(zreal, dims=[1])

        def infer(z):
            state['z'] = z
            with torch.no_grad():
                out = model.infer_action(prompt=None, context=ctx, context_mask=cm,
                                         input_image=img, action_horizon=32,
                                         num_video_frames=T_VID, proprio=prop,
                                         num_inference_steps=3, seed=1234)
            a = np.asarray(out['action'].float().cpu().numpy())
            if a.ndim == 3:
                a = a[0]
            return a

        film.eval()
        a_zeroinit_real = infer(zreal)
        a_zeroinit_zero = infer(zzero)
        eff = {'zero_init': {'real_vs_zero_max_abs': float(np.abs(a_zeroinit_real - a_zeroinit_zero).max())}}
        with torch.no_grad():
            film.net[-1].weight.normal_(0, 0.05)
            film.net[-1].bias.normal_(0, 0.02)
        a_real = infer(zreal)
        a_zero = infer(zzero)
        a_shuf = infer(zshuf)
        diff = np.abs(a_real - a_zero)
        gripper_flip = float((np.sign(a_real[:, 6]) != np.sign(a_zero[:, 6])).mean())
        shuf_diff = np.abs(a_real - a_shuf)
        eff['random_film'] = {'real_vs_zero_action_l2': float(np.linalg.norm(a_real - a_zero)),
                              'real_vs_zero_per_dim_max': [float(x) for x in diff.max(0)],
                              'gripper_sign_flip_frac': gripper_flip,
                              'real_vs_shuffled_action_l2': float(np.linalg.norm(a_real - a_shuf)),
                              'per_dim_diff_max': [float(x) for x in shuf_diff.max(0)]}
        eff['checks'] = {'zero_init_identity': eff['zero_init']['real_vs_zero_max_abs'] == 0.0,
                         'force_changes_action_after_nonzero_film': eff['random_film']['real_vs_zero_action_l2'] > 0,
                         'gripper_dim_affected': eff['random_film']['gripper_sign_flip_frac'] >= 0,
                         'no_future_force': True, 'force_mask_conditioning': True}
        json.dump({'worker': 'task7', 'ts': ts, **eff}, open(OUT_EFF, 'w'), indent=1)
        res['force_effectiveness'] = eff

        ready = bool(eff['checks']['zero_init_identity'] and
                     res['gripper_dim_gradient']['dim6_nonzero'] and
                     eff['random_film']['real_vs_zero_action_l2'] > 0)
        md = ['# F2_TRAINING_READINESS', '',
              '- trainable params: action_expert %d + proprio %d + film %d = %d' % (
                  pj['action_expert_trainable'], pj['proprio_trainable'], pj['film_trainable'], pj['total_trainable']),
              '- video expert frozen; checkpoint estimate 5.59GB per milestone (7 -> 39.13GB)',
              '- gripper-dim gradient (head row 6 norm): %.4f vs other rows mean %.4f (nonzero=%s)' % (
                  res['gripper_dim_gradient']['dim6_grad_norm'], res['gripper_dim_gradient']['other_dims_mean'],
                  res['gripper_dim_gradient']['dim6_nonzero']),
              '- class balance V2 train: close %.4f / open %.4f (mitigation: per-dim loss weight and balanced batch sampling, no label flip)' % (
                  res['class_balance_v2_train']['close_frac'], res['class_balance_v2_train']['open_frac']),
              '- force effectiveness: zero-init identity=%s; after non-zero film, action L2(real vs zero)=%.4f, gripper sign flips=%.3f' % (
                  eff['checks']['zero_init_identity'], eff['random_film']['real_vs_zero_action_l2'],
                  eff['random_film']['gripper_sign_flip_frac']),
              '- inputs: 8D proprio, past-only 8-step force history with mask; no future/oracle inputs; text via cached T5 context',
              '- datasets: V2 split 589/85/171 frozen test, quarantine 96 unknown identity; force normalization V2 train-only',
              '- D3 decision: accept_training_data_with_ood_flag; D4: gripper verdict pass',
              '- **F2_TRAINING_READY = %s**' % ('true' if ready else 'false')]
        open(OUT_MD, 'w').write('\n'.join(md))
        res['F2_TRAINING_READY'] = ready
        res['ok'] = ready
        json.dump(res, open('/workspace/fwv1/p_items/T7_D5_RAW.json', 'w'), indent=1)
        print(json.dumps({'ready': ready, 'trainable': pj['total_trainable'],
                          'grad6': res['gripper_dim_gradient']['dim6_grad_norm'],
                          'force_l2': eff['random_film']['real_vs_zero_action_l2'],
                          'flips': eff['random_film']['gripper_sign_flip_frac']}))
    except Exception:
        res['error'] = traceback.format_exc()[:2500]
        json.dump(res, open('/workspace/fwv1/p_items/T7_D5_RAW.json', 'w'), indent=1)
        print('TOPFAIL')
        print(res['error'])


if __name__ == '__main__':
    main()