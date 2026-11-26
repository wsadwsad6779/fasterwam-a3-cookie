#!/usr/bin/env python3
"""P1 denoise frozen-observation sweep: on FIXED expert observations (esp. GT-closed phases),
run infer_action at K=5/10/20 with 3 PAIRED noise seeds, measure whether larger K raises
closure recall (pred<0.30) and whether it raises false-close on GT-open. No training."""
import importlib.util
import json
import os
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace')
import numpy as np
import torch

CKPT = '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt'
OUT = '/workspace/fwv1/p_items/DENOISE_FROZEN_OBS_RESULTS.json'
KS = [5, 10, 20]
NOISES = [12345, 67890, 24680]   # paired across K
N_BATCH = 10
B = 2
SEED = 20261125
PHASES = ['APPROACH_OPEN', 'CONTACT_CLOSE', 'CARRY_CLOSED', 'DESCEND_CLOSED', 'RELEASE_OPEN', 'RESET_OPEN']


def phase_of(l0, force_last):
    if l0 >= 0.8:
        return 'APPROACH_OPEN' if force_last < 0.2 else 'RELEASE_OPEN'
    if l0 <= 0.30:
        return 'DESCEND_CLOSED' if force_last < 0.2 else 'CARRY_CLOSED'
    return 'CONTACT_CLOSE' if force_last < 0.2 else 'RELEASE_OPEN'


def main():
    import importlib.util as iu
    spec = iu.spec_from_file_location('t7ev', '/workspace/_t7_eval.py')
    ev = iu.module_from_spec(spec); spec.loader.exec_module(ev)
    from t7_data import A3BatchSource
    st = json.load(open('/workspace/fwv1/data/a3_fw_845/dataset_stats_v3.json'))
    a = st['action']['default']
    amin = np.asarray(a['global_min'], np.float32); amax = np.asarray(a['global_max'], np.float32)
    scale = 2.0 / np.where((amax - amin) < 1e-4, 2.0, (amax - amin)); off = -1.0 - scale * amin
    model, film, state = ev.build_model(CKPT)
    src = A3BatchSource(split='val', batch_size=B, seed=SEED, balanced=False)
    # collect frozen observations
    obs_list = []
    for nb in range(N_BATCH):
        bb = src.next_batch()
        for i in range(B):
            l0 = float(np.asarray(bb['grip_raw'][i].detach().cpu().numpy()).astype(np.float32)[0])
            fz = bb['force_history'][i].detach().cpu().numpy()
            fl = float(np.mean(fz[-2:, :2])) if fz.shape[0] else 0.0
            obs_list.append({
                'img': bb['video'][i, :, 0].float().cuda(),
                'prop': bb['proprio'][i, 0].float()[None].cuda(),
                'ctx': bb['context'][i][None].cuda(),
                'mask': bb['context_mask'][i][None].cuda(),
                'z': bb['force_history'][i].reshape(1, -1).float().cuda(),
                'l0': l0, 'phase': phase_of(l0, fl), 'force_last': fl,
            })
    print('collected', len(obs_list), 'obs; phases', {p: sum(1 for o in obs_list if o['phase'] == p) for p in PHASES}, flush=True)

    records = []
    for si, o in enumerate(obs_list):
        state['z'] = o['z']
        for K in KS:
            for nz in NOISES:
                t0 = time.time()
                with torch.no_grad():
                    out = model.infer_action(prompt=None, context=o['ctx'], context_mask=o['mask'],
                                             input_image=o['img'][None], action_horizon=32,
                                             num_video_frames=9, proprio=o['prop'],
                                             num_inference_steps=K, seed=int(nz))
                an = np.asarray(out['action'].float().cpu().numpy()).reshape(32, 7)
                dn = (an - off) / scale
                g = dn[:, 6]
                records.append({'si': si, 'phase': o['phase'], 'gt0': o['l0'], 'K': K, 'noise': nz,
                                'p0': float(g[0]), 'p9': float(g[9]), 'p10': float(g[10]),
                                'p15': float(g[15]), 'p31': float(g[31]),
                                'frac_lt030': float((g < 0.30).mean()), 'infer_ms': round((time.time() - t0) * 1000, 1)})
        if si % 4 == 0:
            print('sample', si, o['phase'], flush=True)

    # aggregate per phase per K
    agg = {}
    for ph in PHASES:
        sel = [r for r in records if r['phase'] == ph]
        if not sel:
            agg[ph] = {'n_samples': 0}
            continue
        gtclosed = [r for r in sel if r['gt0'] <= 0.30]
        gtopens = [r for r in sel if r['gt0'] >= 0.8]
        row = {'n_samples': len({r['si'] for r in sel}), 'n_records': len(sel)} 
        for K in KS:
            kk = [r for r in sel if r['K'] == K]
            rec = {'n': len(kk),
                   'pred0_median': float(np.median([r['p0'] for r in kk])) if kk else None,
                   'pred0_lt030_frac': float(np.mean([r['p0'] < 0.30 for r in kk])) if kk else None,
                   'pred31_lt030_frac': float(np.mean([r['p31'] < 0.30 for r in kk])) if kk else None,
                   'frac_lt030_mean': float(np.mean([r['frac_lt030'] for r in kk])) if kk else None,
                   'infer_ms_median': float(np.median([r['infer_ms'] for r in kk])) if kk else None}
            if gtclosed:
                kc = [r for r in gtclosed if r['K'] == K]
                rec['gtclosed_pred0_lt030_recall'] = float(np.mean([r['p0'] < 0.30 for r in kc])) if kc else None
                rec['gtclosed_pred31_lt030'] = float(np.mean([r['p31'] < 0.30 for r in kc])) if kc else None
            if gtopens:
                ko = [r for r in gtopens if r['K'] == K]
                rec['gtopen_falseclose_pred0_lt030'] = float(np.mean([r['p0'] < 0.30 for r in ko])) if ko else None
            row['K%d' % K] = rec
        agg[ph] = row

    od = {'artifact': 'DENOISE_FROZEN_OBS_RESULTS', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'checkpoint': CKPT, 'checkpoint_sha256': ev.sha256_file(CKPT),
          'Ks': KS, 'noises_paired': NOISES, 'n_obs': len(obs_list),
          'protocol': 'frozen expert observations from val; same img/prop/z; only K & paired noise vary; r=H not used (teacher-forced single infer)',
          'per_phase': agg, 'records': records,
          'note': 'tests H1: does larger K raise closure recall on GT-closed expert frames (without raising false-close on GT-open)?'}
    json.dump(od, open(OUT, 'w'), indent=1)
    print(json.dumps({'DONE': True, 'per_phase_keys': {p: agg[p].get('n_samples') for p in PHASES}}))


if __name__ == '__main__':
    main()
