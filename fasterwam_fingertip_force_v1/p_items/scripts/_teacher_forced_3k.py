import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, '/workspace')
from t7_data import A3BatchSource  # noqa: E402

CKPT = os.environ.get('TF_CKPT', '/root/fwv1_checkpoint_archive/step_003000.pt')
STATS = '/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'
OUT = os.environ.get('TF_OUT', '/workspace/fwv1/p_items/TEACHER_FORCED_3K.json')
N_BATCH = int(os.environ.get('T7_TF_BATCHES', '4'))
B = 2
SEED = 20261125

spec = importlib.util.spec_from_file_location('t7ev', '/workspace/_t7_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

st = json.load(open(STATS))
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
        if labels[0] >= 0.8:
            phase = 'APPROACH_OPEN' if force_last < 0.2 else 'RESET_OPEN'
        elif labels[0] <= 0.30:
            phase = 'CARRY_CLOSED' if force_last >= 0.2 else 'DESCEND_CLOSED'
        else:
            phase = 'RELEASE_OPEN' if force_last >= 0.2 else 'CONTACT_CLOSE'
        row = {'batch': nb, 'i': i, 'phase': phase, 'force_z_last': force_last,
               'label': [float(x) for x in labels],
               'pred': {}}
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
            row['pred'][variant] = [float(x) for x in dn[:, 6]]
        samples.append(row)
    print('batch', nb, 'done', flush=True)

summary = {}
for variant in ('normal', 'zero', 'skip'):
    per = {}
    for ph in sorted({s['phase'] for s in samples}):
        sel = [s for s in samples if s['phase'] == ph]
        if not sel:
            continue
        p0 = np.array([s['pred'][variant][0] for s in sel])
        p9 = np.array([s['pred'][variant][9] for s in sel])
        p10 = np.array([s['pred'][variant][10] for s in sel])
        p31 = np.array([s['pred'][variant][31] for s in sel])
        l0 = np.array([s['label'][0] for s in sel])
        per[ph] = {'n': len(sel),
                   'label0_lt_0.30': float((l0 < 0.30).mean()),
                   'pred0_lt_0.30': float((p0 < 0.30).mean()),
                   'pred9_lt_0.30': float((p9 < 0.30).mean()),
                   'pred10_lt_0.30': float((p10 < 0.30).mean()),
                   'pred31_lt_0.30': float((p31 < 0.30).mean()),
                   'pred0_median': float(np.median(p0)),
                   'pred9_median': float(np.median(p9)),
                   'pred31_median': float(np.median(p31)),
                   'label0_median': float(np.median(l0))}
    summary[variant] = per
json.dump({'samples': samples, 'summary': summary, 'n_samples': len(samples),
           'checkpoint': CKPT, 'protocol': 'expert validation observations, 32-step infer, 5 denoise steps'},
          open(OUT, 'w'), indent=1)
print(json.dumps({'n': len(samples), 'summary': summary}))