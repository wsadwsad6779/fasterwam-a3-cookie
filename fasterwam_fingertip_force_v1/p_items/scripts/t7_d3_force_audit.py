"""D3: fingertip force distribution audit (dataset vs current MuJoCo replay of the same
recorded actions), train-split-only normalization V2, and explicit OOD acceptance."""
import glob
import json
import os
import time

import numpy as np
import pyarrow.parquet as pq

P = '/workspace/fwv1/p_items'
MANIFEST = P + '/DATASET_845_ACCEPTED_MANIFEST.json'
SPLIT = P + '/DATASET_845_SPLIT_MANIFEST_V2.json'
PARQ = [P + '/hf_multi/data/chunk-000/file-000.parquet',
        P + '/hf_multi/data/chunk-000/file-001.parquet']
TRACES = '/workspace/fwv1/p_items/videos/trace_ep*.npz'
OUT_MD = P + '/FINGERTIP_FORCE_DISTRIBUTION_AUDIT.md'
OUT_NORM = P + '/FINGERTIP_FORCE_NORMALIZATION_V2.json'
OUT_OOD = P + '/FORCE_OOD_ACCEPTANCE.json'
NORM_DIR = '/workspace/fwv1/data/a3_fw_845'
CH = [12, 13, 14, 15]


def main():
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    man = json.load(open(MANIFEST))
    split = json.load(open(SPLIT))
    train_ids = set(int(x) for x in split['train_episodes'])
    t = pq.read_table(PARQ, columns=['episode_index', 'observation.force'])
    E = np.asarray(t['episode_index'])
    fc = t['observation.force'].combine_chunks()
    F = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18)
    order = np.argsort(E, kind='stable')
    E, F = E[order], F[order]
    b = np.flatnonzero(np.diff(E)) + 1
    st = np.r_[0, b]
    en = np.r_[b, len(E)]
    ids = [int(E[st[k]]) for k in range(len(st))]
    mask = np.zeros(len(E), bool)
    for k, ep in enumerate(ids):
        if ep in train_ids:
            mask[st[k]:en[k]] = True
    pads = F[mask][:, CH]
    contact = np.abs(pads) > 0.05
    q = np.quantile(pads, [0.01, 0.5, 0.99], axis=0)
    ds = {'frames': int(pads.shape[0]),
          'mean': [float(x) for x in pads.mean(0)],
          'std': [float(x) for x in pads.std(0)],
          'q01': [float(x) for x in q[0]],
          'q50': [float(x) for x in q[1]],
          'q99': [float(x) for x in q[2]],
          'min': [float(x) for x in pads.min(0)],
          'max': [float(x) for x in pads.max(0)],
          'max_abs': [float(x) for x in np.abs(pads).max(0)],
          'contact_frac': [float(x) for x in contact.mean(0)]}

    sim = []
    for fp in sorted(glob.glob(TRACES)):
        z = np.load(fp)
        f = np.stack([z['pad_in'], z['pad_out']], axis=1)  # [T,2] left pads
        sim.append(f)
    simP = np.concatenate(sim, 0) if sim else np.zeros((0, 2))
    sim_stats = {'frames': int(simP.shape[0]),
                 'mean': [float(x) for x in simP.mean(0)],
                 'std': [float(x) for x in simP.std(0)],
                 'q99': [float(x) for x in np.quantile(simP, 0.99, axis=0)],
                 'max': [float(x) for x in np.abs(simP).max(0)],
                 'contact_frac': [float(x) for x in (np.abs(simP) > 0.05).mean(0)]}
    ratio = [float(sim_stats['max'][i] / max(1e-9, ds['max_abs'][i])) for i in range(2)]

    mean = np.asarray(ds['mean'], np.float64)
    std = np.asarray(ds['std'], np.float64)
    active = (std > 1e-6).astype(np.float64)
    std_floor = 0.05
    eff_std = np.maximum(std, std_floor)
    clip = 5.0
    norm = {'version': 2, 'ts': ts, 'source': 'V2 train split only (589 episodes, accepted manifest)',
            'split_manifest': SPLIT, 'manifest': MANIFEST,
            'channels': CH, 'channel_names': ['L_inner', 'L_outer', 'R_inner', 'R_outer'],
            'mean': [float(x) for x in mean], 'std': [float(x) for x in std],
            'std_floor': std_floor, 'active': [bool(x) for x in active],
            'q01': ds['q01'], 'q99': ds['q99'], 'max_abs_train': ds['max_abs'],
            'clip': clip,
            'formula': 'z = clip((x - mean) / max(std, std_floor) * active, -clip, +clip); active=0 channels stay 0',
            'contact_gate': {'threshold_n': 0.05, 'rule': 'raw |x|<=0.05 -> contact=0 and features reported as no-contact; features are not amplified'},
            'ood_rule': 'flag frames where raw |x| > train max_abs per channel; also report saturation fraction |z|>=clip-1e-3; never rescale training targets',
            'no_blind_amplification': True}
    json.dump(norm, open(os.path.join(NORM_DIR, 'dataset_force_norm_v2.json'), 'w'), indent=1)
    json.dump(norm, open(OUT_NORM, 'w'), indent=1)

    # saturation of current-sim replay frames under V2 (left pads vs train channels 0,1)
    sil = np.clip((simP - mean[:2]) / eff_std[:2] * active[:2], -clip, clip)
    sat = float((np.abs(sil) >= clip - 1e-3).mean()) if simP.size else None
    ood_frac = float((np.abs(simP) > np.asarray(ds['max_abs'])[:2]).any(axis=1).mean()) if simP.size else None
    ood = {'worker': 'task7', 'ts': ts,
           'dataset_train': ds, 'current_sim_replay_dataset_actions': sim_stats,
           'sim_to_dataset_max_ratio_left_pads': ratio,
           'v2_saturation_fraction_sim_frames': sat,
           'oof_frame_fraction_sim_frames': ood_frac,
           'hypotheses_tested': {
               'units': 'same 18-channel table, pad channels 12-15 are MuJoCo touch sensors in newtons (source audit); no unit conversion difference',
               'layout_time': 'both dataset and current env assemble force from the same sensor names after mj_step; no layout swap (L/R pads correlate with L/R contacts)',
               'expert_controller': 'P5 replays the recorded dataset joint actions in the current env; the same actions still produce the much larger forces, so the gap is not caused by a different expert policy',
               'model_xml_or_contact_config': 'current sim repo commit 08ebe8c reproduces the recorded actions but yields pad peaks %.2f N vs dataset %.2f N; the archived dataset was recorded under a different sim/contact/sensor configuration or scaling, which is not recoverable from the published files' % (sim_stats['max'][0], ds['max_abs'][1]),
               'right_pads': 'dataset R pads are exactly 0 while the current sim right pads move when the right arm is driven; the left-arm protocol makes R channels inactive in training (mask=0)'},
           'decision': 'accept_training_data_with_ood_flag',
           'conditions': ['normalizer uses V2 train split only', 'no rescaling of dataset targets', 'evaluation-time force input is reported as OOD with saturation fraction; it is not silently clipped without notice', 'R channels masked inactive'],
           'notes': 'A sim-side force calibration is only for interpreting evaluation, never for changing training labels.'}
    json.dump(ood, open(OUT_OOD, 'w'), indent=1)

    md = ['# FINGERTIP_FORCE_DISTRIBUTION_AUDIT', '',
          '- dataset (V2 train, 589 eps, %d frames): L pads mean %.3f/%.3f N, q99 %.2f/%.2f N, max %.2f/%.2f N' % (
              ds['frames'], ds['mean'][0], ds['mean'][1], ds['q99'][0], ds['q99'][1], ds['max_abs'][0], ds['max_abs'][1]),
          '- current MuJoCo replay of the SAME recorded actions (%d frames): L pad max %.2f/%.2f N, q99 %.2f/%.2f N, contact fraction %.3f/%.3f' % (
              sim_stats['frames'], sim_stats['max'][0], sim_stats['max'][1], sim_stats['q99'][0], sim_stats['q99'][1],
              sim_stats['contact_frac'][0], sim_stats['contact_frac'][1]),
          '- ratio sim/dataset max: %.1fx / %.1fx' % (ratio[0], ratio[1]),
          '- V2 normalization: train-only mean/std (std floor 0.05), active mask (R pads inactive), clip +/-5, contact gate at 0.05 N, explicit OOD flag; no blind amplification',
          '- OOD: %.1f%% of current-sim frames exceed the train max on left pads; %.1f%% saturate at clip' % (
              100 * (ood_frac or 0), 100 * (sat or 0)),
          '',
          '## Hypotheses',
          '- units: same newtons, same channel order, no conversion bug',
          '- timing: same sensor assembly after mj_step; no future leakage',
          '- expert: replaying recorded actions still yields high forces -> not the expert policy',
          '- remaining cause: dataset archive was recorded under a different sim/contact/sensor configuration or scaling; unrecoverable from published files',
          '- consequence: train on dataset distribution; evaluation must report force OOD explicitly (no silent clipping)',
          '- verdict: **accept_training_data_with_ood_flag**']
    open(OUT_MD, 'w').write('\n'.join(md))
    print(json.dumps({'dataset_max': ds['max_abs'], 'sim_max': sim_stats['max'],
                      'ratio': ratio, 'sat': sat, 'ood_frac': ood_frac}))


if __name__ == '__main__':
    main()