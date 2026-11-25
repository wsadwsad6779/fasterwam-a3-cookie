#!/usr/bin/env python3
"""B3: expanded force ablation. Teacher-forced closure prediction on val episodes under
normal / force-zero / skip-FiLM. Reports per-phase closure and the normal-minus-ablation deltas
to attribute the force-input effect. F2 15k read-only; no training."""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/workspace')

CKPT = os.environ.get('B3_CKPT', '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt')
OUT = os.environ.get('B3_OUT', '/workspace/fwv1/p_items/B3_FORCE_ABLATION_EXPANDED.json')
N_BATCH = int(os.environ.get('B3_BATCHES', '10'))
B = 2
SEED = 20261125
PHASES = ['APPROACH', 'CONTACT', 'CLOSE', 'CARRY', 'TRANSPORT', 'RELEASE']


def phase_of(l0, force_last):
    if force_last < 0.2 and l0 >= 0.8:
        return 'APPROACH'
    if force_last < 0.2 and 0.30 < l0 < 0.8:
        return 'CONTACT'
    if force_last < 0.2 and l0 <= 0.30:
        return 'CLOSE'
    if force_last >= 0.2 and l0 <= 0.30:
        return 'CARRY'
    if force_last >= 0.2 and 0.30 < l0 < 0.8:
        return 'TRANSPORT'
    return 'RELEASE'


def main():
    from t7_data import A3BatchSource
    spec = importlib.util.spec_from_file_location('t7ev', '/workspace/_t7_eval.py')
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    st = json.load(open('/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'))
    a = st['action']['default']
    amin = np.asarray(a['global_min'], np.float32)
    amax = np.asarray(a['global_max'], np.float32)
    scale = 2.0 / np.where((amax - amin) < 1e-4, 2.0, (amax - amin))
    off = -1.0 - scale * amin
    model, film, state = ev.build_model(CKPT)
    src = A3BatchSource(split='val', batch_size=B, seed=SEED, balanced=False)
    samples = []
    for nb in range(N_BATCH):
        bb = src.next_batch()
        for i in range(B):
            img = bb['video'][i, :, 0].float().cuda()
            prop = bb['proprio'][i, 0].float()[None].cuda()
            ctx = bb['context'][i][None].cuda()
            mask = bb['context_mask'][i][None].cuda()
            z = bb['force_history'][i].reshape(1, -1).float().cuda()
            labels = bb['grip_raw'][i].detach().cpu().numpy().astype(np.float32)
            fz = bb['force_history'][i].detach().cpu().numpy()
            force_last = float(np.mean(fz[-2:, :2])) if fz.shape[0] else 0.0
            l0 = float(labels[0])
            preds = {}
            for variant in ('normal', 'zero', 'skip'):
                if variant == 'skip':
                    state['z'] = None
                elif variant == 'zero':
                    state['z'] = torch.zeros_like(z)
                else:
                    state['z'] = z
                with torch.no_grad():
                    out = model.infer_action(prompt=None, context=ctx, context_mask=mask,
                                             input_image=img[None], action_horizon=32,
                                             num_video_frames=9, proprio=prop,
                                             num_inference_steps=5, seed=SEED + nb * 10 + i)
                an = np.asarray(out['action'].float().cpu().numpy()).reshape(32, 7)
                dn = (an - off) / scale
                preds[variant] = {'p0': float(dn[0, 6]), 'p9': float(dn[9, 6]),
                                  'p10': float(dn[10, 6]), 'p31': float(dn[31, 6])}
            samples.append({'batch': nb, 'i': i, 'phase': phase_of(l0, force_last),
                            'label0': l0, 'force_last': force_last, 'preds': preds})
        print('B3 batch %d' % nb, flush=True)
    per = {}
    for ph in PHASES:
        sel = [s for s in samples if s['phase'] == ph]
        if not sel:
            per[ph] = {'n': 0}
            continue
        l0 = np.array([s['label0'] for s in sel])
        row = {'n': len(sel), 'label0_lt_0.30': float((l0 < 0.30).mean()), 'label0_median': float(np.median(l0))}
        for variant in ('normal', 'zero', 'skip'):
            for pos in ('p0', 'p9', 'p10', 'p31'):
                v = np.array([s['preds'][variant][pos] for s in sel])
                row['%s_%s_lt_0.30' % (variant, pos)] = float((v < 0.30).mean())
                row['%s_%s_median' % (variant, pos)] = float(np.median(v))
        for variant in ('zero', 'skip'):
            row['delta_p0_%s_minus_normal' % variant] = row['%s_p0_median' % variant] - row['normal_p0_median']
            row['delta_p31_%s_minus_normal' % variant] = row['%s_p31_median' % variant] - row['normal_p31_median']
        per[ph] = row
    json.dump({'artifact': 'B3_FORCE_ABLATION_EXPANDED',
               'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'checkpoint': CKPT, 'n_samples': len(samples), 'per_phase': per,
               'note': 'teacher-forced normal/zero/skip ablation; no training; deltas = ablation minus normal (median grip)'},
              open(OUT, 'w'), indent=1)
    print(json.dumps({'B3_DONE': True, 'n': len(samples)}), flush=True)


if __name__ == '__main__':
    main()
