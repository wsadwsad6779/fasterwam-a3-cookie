"""Gripper chatter vs state: seed-ensemble sweep at real replan boundaries.

Why
---
At f=440 the gripper sign sequence is nearly independent of the initial noise (per-row
agreement ~0.5 across seeds) and 50 Euler steps do not remove the chatter, while f=80 is
stable at 0 flips for 10 and 50 steps. That rules out the discretisation as the cause and
leaves the conditional distribution itself. This sweep asks where along the episode the
distribution becomes chattery and whether that tracks the state's distance from the
demonstration support (pose height / force / opening motion).

Design (predeclared, bounded: 9 states x 5 seeds = 45 single-action inferences, ~1.5 s each)
  states  f = 0, 30, 50, 80, 110, 140, 200, 300, 440
  seeds   protocol noise_seed(17, k+1) and 1001..1004
  metric  flips_32, flips_first10, per-row sign agreement across seeds, |mean value| per row,
          and the state descriptors (eef pose, force absmax and per-channel sigma, opening,
          opening rate, chunk index)

Predictions / falsifiers
  H_label : chatter already exists in-distribution (f=0/30/50) -> supervision/loss property.
            falsified if every in-distribution state has 0 flips in all 5 seeds.
  H_state : chatter grows with state distance from the demo support (pose z, force sigma).
            falsified if chatter is flat in state descriptors.
The f=0 protocol seed doubles as an identity gate against recorded chunk 0.
"""
import copy
import json
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu_gripper_state_sweep_20261118'
OUT.mkdir(exist_ok=False)
OBS_NPZ = ROOT / 'forensics_full_wide_out' / 'seed17_observations.npz'
CHUNK_NPZ = ROOT / 'forensics_full_wide_out' / 'seed17_original_chunks.npz'
SEED = 17
STATES = [0, 30, 50, 80, 110, 140, 200, 300, 440]
ALT_SEEDS = [1001, 1002, 1003, 1004]

E, factory, W, fw = D.setup_protocol()
import torch
from force_stream import ForceStream, CHANNELS, CHANNEL_NAMES
from policy40k import noise_seed

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training

OBS = np.load(OBS_NPZ, allow_pickle=False)
CHUNKS = np.load(CHUNK_NPZ, allow_pickle=False)
CAL = json.loads((ROOT.parent / 'step_040000' / 'force_calib_v1.json').read_text())
MEAN = np.asarray(CAL['mean'], np.float64)
STD = np.asarray(CAL['std'], np.float64)
FCH = [int(c) for c in CAL['channels']]
opening_obs = np.asarray(OBS['observation.state'], np.float64)[:, 7]
eef_obs = np.asarray(OBS['observation.eef_pose'], np.float64)
force_obs = np.asarray(OBS['observation.force'], np.float64)

corpus = np.load(ROOT.parent / 'archive_evidence' / 'data' / 'a3_fw' / 'arrays.npz',
                 allow_pickle=True)
prop = np.asarray(corpus['proprio'], np.float64)
support = {'x': [float(prop[:, 0].min()), float(prop[:, 0].max())],
           'y': [float(prop[:, 1].min()), float(prop[:, 1].max())],
           'z': [float(prop[:, 2].min()), float(prop[:, 2].max())],
           'z_p99': float(np.percentile(prop[:, 2], 99)),
           'z_p999': float(np.percentile(prop[:, 2], 99.9))}
(OUT / 'identity.json').write_text(json.dumps({
    **D.MARKERS, 'source_sha256': D.sha256_file(Path(__file__)),
    'protocol': facts, 'states': STATES, 'alt_seeds': ALT_SEEDS,
    'corpus_support': support,
    'channel_names': list(CHANNEL_NAMES),
    'scope': 'offline single-action inference at recorded replan states, no control executed',
}))


def infer(frame, ns):
    obs = {k: OBS[k][frame] for k in OBS.files if k.startswith('observation.')}
    stream = ForceStream()
    for f in range(max(0, frame - 15), frame + 1):
        stream.push(OBS['observation.force'][f])
    history, mask = stream.snapshot()
    kw = policy._infer_kwargs(obs, history, mask, ns)
    random.seed(9081)
    np.random.seed(9081)
    torch.manual_seed(9081)
    torch.cuda.manual_seed_all(9081)
    with torch.no_grad():
        pred = policy.model.infer_action(**copy.deepcopy(kw))
    return pred['action'].detach().float().cpu().numpy().reshape(32, 7)


summary = []
for frame in STATES:
    proto = noise_seed(SEED, frame // 10 + 1)
    seeds = [proto] + ALT_SEEDS
    raws = []
    t0 = time.time()
    for ns in seeds:
        raws.append(infer(frame, ns))
    wall = time.time() - t0
    stack = np.stack(raws)                      # (S, 32, 7)
    v = stack[:, :, 6]
    s = np.where(v >= 0, 1, -1)
    flips = [int((np.diff(x) != 0).sum()) for x in s]
    flips10 = [int((np.diff(x[:10]) != 0).sum()) for x in s]
    agree = [(s[:, t] == s[0, t]).mean() for t in range(32)]
    mean_v = v.mean(0)
    ens = np.where(mean_v >= 0, 1, -1)
    fmax = float(np.abs(force_obs[frame, FCH]).max())
    z_force = float(np.abs((force_obs[frame, FCH] - MEAN) / STD).max())
    dprev = float(opening_obs[frame] - opening_obs[frame - 1]) if frame > 0 else 0.0
    rec = {'frame': frame, 'chunk': frame // 10, 'protocol_seed': int(proto),
           'seeds': seeds, 'wall_s': round(wall, 2),
           'flips_per_seed': flips, 'flips_first10_per_seed': flips10,
           'flips_mean': float(np.mean(flips)), 'flips_min': int(np.min(flips)),
           'flips_max': int(np.max(flips)),
           'ensemble_median_flips': int((np.diff(ens) != 0).sum()),
           'mean_row_agreement': float(np.mean(agree)),
           'min_row_agreement': float(np.min(agree)),
           'frac_rows_majority': float((np.abs(mean_v) > 1e-6).mean()),
           'abs_mean_value_first10': np.abs(mean_v[:10]).round(4).tolist(),
           'eef_xyz': eef_obs[frame, :3].round(4).tolist(),
           'eef_z': float(eef_obs[frame, 2]),
           'opening': float(opening_obs[frame]),
           'd_opening_prev': dprev,
           'force_absmax_N': fmax, 'force_max_abs_sigma': z_force}
    summary.append(rec)
    np.savez_compressed(OUT / ('f%03d_raw32.npz' % frame), raw32=stack.astype(np.float32),
                        seeds=np.asarray(seeds), **D.MARKERS)
    # identity gate at f0 (chunk 0)
    if frame == 0:
        ref = np.asarray(CHUNKS['chunk_000_model_out'], np.float64).reshape(32, 7)
        gate = {'max_abs_diff': float(np.abs(raws[0] - ref).max()),
                'bit_equal': bool(np.array_equal(raws[0], ref))}
        rec['gate'] = gate
        print('[gate f0]', json.dumps(gate), flush=True)
    print('[f%3d] flips %s mean %.2f ens %d agree %.2f z=%.3f fmax=%.1f sigma=%.1f'
          % (frame, flips, rec['flips_mean'], rec['ensemble_median_flips'],
             rec['mean_row_agreement'], rec['eef_z'], fmax, z_force), flush=True)
    (OUT / 'summary.json').write_text(json.dumps({**D.MARKERS, 'complete': False,
                                                  'results': summary}, indent=2, default=str))

# relation to state distance
z = np.array([r['eef_z'] for r in summary])
sig = np.array([r['force_max_abs_sigma'] for r in summary])
fm = np.array([r['flips_mean'] for r in summary])
ag = np.array([r['mean_row_agreement'] for r in summary])
rel = {}
if len(summary) > 2:
    rel = {'corr_flips_vs_eef_z': float(np.corrcoef(fm, z)[0, 1]),
           'corr_flips_vs_force_sigma': float(np.corrcoef(fm, sig)[0, 1]),
           'corr_agreement_vs_eef_z': float(np.corrcoef(ag, z)[0, 1]),
           'corr_agreement_vs_force_sigma': float(np.corrcoef(ag, sig)[0, 1])}
(OUT / 'summary.json').write_text(json.dumps({**D.MARKERS, 'complete': True,
                                              'states': STATES, 'corpus_support': support,
                                              'relations': rel, 'results': summary},
                                             indent=2, default=str))
print('RELATIONS', json.dumps(rel), flush=True)
print('ALL_DONE', flush=True)