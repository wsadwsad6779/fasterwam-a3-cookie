#!/usr/bin/env python3
"""B2: six-phase teacher-forced analysis across independent val episodes (V7 phase names).
Phases (V7): APPROACH_OPEN / CONTACT_CLOSE / CARRY_CLOSED / DESCEND_CLOSED / RELEASE_OPEN / RESET_OPEN.
Feeds expert observations to F2 15k (read-only), measures per-phase closure prediction.
No training. Records per-sample episode id to prove cross-episode coverage."""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, '/workspace')

CKPT = os.environ.get('B2_CKPT', '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt')
OUT = os.environ.get('B2_OUT', '/workspace/fwv1/p_items/B2_SIXPHASE_TEACHER.json')
N_BATCH = int(os.environ.get('B2_BATCHES', '12'))
B = 2
SEED = 20261125
V7 = ['APPROACH_OPEN', 'CONTACT_CLOSE', 'CARRY_CLOSED', 'DESCEND_CLOSED', 'RELEASE_OPEN', 'RESET_OPEN']
# explicit mapping rule from the two observable signals (grip label0, fingertip force) to V7 names
MAPPING = ("open(label0>=0.8)&force<0.2->APPROACH_OPEN; "
           "open&force>=0.2->RELEASE_OPEN; "
           "closed(label0<=0.30)&force<0.2->DESCEND_CLOSED; "
           "closed&force>=0.2->CARRY_CLOSED; "
           "mid(0.30<label0<0.8)&force<0.2->CONTACT_CLOSE; "
           "mid&force>=0.2->RELEASE_OPEN; "
           "RESET_OPEN not separable from (label0,force) alone -> reported n=0")


def v7_phase(l0, force_last):
    if l0 >= 0.8:
        return 'APPROACH_OPEN' if force_last < 0.2 else 'RELEASE_OPEN'
    if l0 <= 0.30:
        return 'DESCEND_CLOSED' if force_last < 0.2 else 'CARRY_CLOSED'
    return 'CONTACT_CLOSE' if force_last < 0.2 else 'RELEASE_OPEN'


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
    episodes = set()
    for nb in range(N_BATCH):
        bb = src.next_batch()
        ep = (bb.get('meta') or {}).get('episode')
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
            state['z'] = z
            with torch.no_grad():
                out = model.infer_action(prompt=None, context=ctx, context_mask=mask,
                                         input_image=img[None], action_horizon=32,
                                         num_video_frames=9, proprio=prop,
                                         num_inference_steps=5, seed=SEED + nb * 10 + i)
            an = np.asarray(out['action'].float().cpu().numpy()).reshape(32, 7)
            dn = (an - off) / scale
            samples.append({'batch': nb, 'i': i, 'episode': (int(ep) if isinstance(ep, (int, np.integer)) else ep),
                            'phase': v7_phase(l0, force_last), 'label0': l0, 'force_last': force_last,
                            'pred0': float(dn[0, 6]), 'pred9': float(dn[9, 6]),
                            'pred10': float(dn[10, 6]), 'pred31': float(dn[31, 6])})
            if ep is not None:
                episodes.add(int(ep) if isinstance(ep, (int, np.integer)) else str(ep))
        print('B2 batch %d episodes=%d' % (nb, len(episodes)), flush=True)
    per = {}
    for ph in V7:
        sel = [s for s in samples if s['phase'] == ph]
        if not sel:
            per[ph] = {'n': 0}
            continue
        l0 = np.array([s['label0'] for s in sel])
        p0 = np.array([s['pred0'] for s in sel])
        p9 = np.array([s['pred9'] for s in sel])
        p10 = np.array([s['pred10'] for s in sel])
        p31 = np.array([s['pred31'] for s in sel])
        per[ph] = {'n': len(sel),
                   'label0_lt_0.30': float((l0 < 0.30).mean()),
                   'pred0_lt_0.30': float((p0 < 0.30).mean()),
                   'pred9_lt_0.30': float((p9 < 0.30).mean()),
                   'pred10_lt_0.30': float((p10 < 0.30).mean()),
                   'pred31_lt_0.30': float((p31 < 0.30).mean()),
                   'label0_median': float(np.median(l0)),
                   'pred0_median': float(np.median(p0)),
                   'pred31_median': float(np.median(p31))}
    json.dump({'artifact': 'B2_SIXPHASE_TEACHER',
               'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'checkpoint': CKPT, 'n_samples': len(samples), 'n_episodes': len(episodes),
               'episodes': sorted(episodes, key=lambda x: str(x)),
               'per_phase': per, 'v7_phase_names': V7, 'v7_phase_rule': MAPPING,
               'per_sample_episode': [{'batch': s['batch'], 'i': s['i'], 'episode': s['episode'], 'phase': s['phase']} for s in samples],
               'note': 'teacher-forced infer_action on val episodes; no training; V7 phase names with explicit mapping'},
              open(OUT, 'w'), indent=1)
    print(json.dumps({'B2_DONE': True, 'n': len(samples), 'episodes': len(episodes)}), flush=True)


if __name__ == '__main__':
    main()
