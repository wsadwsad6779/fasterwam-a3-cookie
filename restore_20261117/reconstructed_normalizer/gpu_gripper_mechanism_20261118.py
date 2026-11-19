"""Gripper mechanism test: sampler step count vs initial-noise seed at fixed real states.

Question
--------
The P2-clamped and original rollouts both show gripper sign chatter that exceeds the
training-label support (labels: max 10 flips per 32-row window over 39,481 windows; the
model reaches 18-19 at chunks 33/35/40/42). Is that chatter produced by
  M1  the 10-step Euler discretisation of the flow ODE, or
  M2  the learned conditional distribution (state-dependent uncertainty), or
  M3  the initial noise draw only?

Design (predeclared, bounded)
-----------------------------
States: f=440 (chatter chunk 44) and f=80 (clean chunk 8), both real replan boundaries.
  A      protocol noise, 10 steps      - identity gate vs the recorded chunk (bit exact)
  B50    protocol noise, 50 steps      - M1 test
  C1..C4 4 other initial-noise seeds, 10 steps   - M3 test
  C1_50  one alternate seed, 50 steps  - interaction

Predictions / falsifiers
------------------------
  M1 supported  : B50 flip count <= 50% of A while the executed majority sign is unchanged.
  M1 rejected   : B50 retains >= 70% of A's flips.
  M3 supported  : C1..C4 flip counts spread widely and per-row sign agreement with A is low.
  M2 supported  : chatter survives both (B50 keeps flips; seeds disagree in pattern but each
                  stays equally chattery).

Nothing is executed in the simulator; no asset is modified.
"""
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'gpu_gripper_mechanism_20261118'
OUT.mkdir(exist_ok=False)
OBS_NPZ = ROOT / 'forensics_full_wide_out' / 'seed17_observations.npz'
CHUNK_NPZ = ROOT / 'forensics_full_wide_out' / 'seed17_original_chunks.npz'
SEED = 17
PROTOCOL_STEPS = 10

E, factory, W, fw = D.setup_protocol()
import torch
from force_stream import ForceStream, CHANNELS, CHANNEL_NAMES
from policy40k import noise_seed

policy, doc, calib, facts = D.build_policy_reconstructed(
    E, W, SimpleNamespace(cpu=False, gripper_mode='pm1'))
assert facts['device'] == 'cuda' and not policy.model.training

OBS = np.load(OBS_NPZ, allow_pickle=False)
CHUNKS = np.load(CHUNK_NPZ, allow_pickle=False)


def flips(s):
    return int((np.diff(s) != 0).sum())


def summarize(raw):
    v = raw[:, 6]
    s = np.where(v >= 0, 1, -1)
    return {
        'flips_32': flips(s),
        'flips_first10': flips(s[:10]),
        'sign10': ''.join('+' if x > 0 else '-' for x in s[:10]),
        'sign32': ''.join('+' if x > 0 else '-' for x in s),
        'absmax': float(np.abs(raw).max()),
        'absmin_first10': float(np.abs(v[:10]).min()),
        'mean_abs': float(np.abs(v).mean()),
    }


def run_job(frame, ns, steps, name):
    obs = {k: OBS[k][frame] for k in OBS.files if k.startswith('observation.')}
    stream = ForceStream()
    for f in range(max(0, frame - 15), frame + 1):
        stream.push(OBS['observation.force'][f])
    history, mask = stream.snapshot()
    kw = policy._infer_kwargs(obs, history, mask, ns)
    assert int(kw['num_inference_steps']) == PROTOCOL_STEPS
    if steps != PROTOCOL_STEPS:
        kw['num_inference_steps'] = steps
    random.seed(9081)
    np.random.seed(9081)
    torch.manual_seed(9081)
    torch.cuda.manual_seed_all(9081)
    t0 = time.time()
    with torch.no_grad():
        pred = policy.model.infer_action(**copy.deepcopy(kw))
    wall = time.time() - t0
    raw = pred['action'].detach().float().cpu().numpy().reshape(32, 7)
    future = pred.get('force_future_newton')
    payload = {'raw32': raw,
               'force_future_newton': (future.detach().float().cpu().numpy()
                                       if future is not None else np.zeros((1, 0))),
               **D.MARKERS}
    np.savez_compressed(OUT / (name + '.npz'), **payload)
    rec = {**D.MARKERS, 'name': name, 'frame': frame, 'noise_seed': int(ns), 'steps': int(steps),
           'wall_s': round(wall, 2), **summarize(raw),
           'npz_sha256': D.sha256_file(OUT / (name + '.npz'))}
    (OUT / (name + '.json')).write_text(json.dumps(rec, indent=2, default=str))
    print('[%s] f%d steps=%d ns=%d flips32=%d flips10=%d sign10=%s wall=%.1fs'
          % (name, frame, steps, ns, rec['flips_32'], rec['flips_first10'], rec['sign10'], wall),
          flush=True)
    return rec


def gate(frame, rec):
    chunk = frame // 10
    ref = np.asarray(CHUNKS['chunk_%03d_model_out' % chunk], np.float64).reshape(32, 7)
    got = np.load(OUT / (rec['name'] + '.npz'))['raw32'].astype(np.float64)
    diff = float(np.abs(got - ref).max())
    out = {'chunk': chunk, 'max_abs_diff': diff, 'bit_equal': bool(np.array_equal(got, ref))}
    print('[gate f%d] %s' % (frame, json.dumps(out)), flush=True)
    return out


JOBS = []
for frame, tag in ((440, 'f440'), (80, 'f80')):
    proto = noise_seed(SEED, frame // 10 + 1)
    JOBS.append({'frame': frame, 'tag': tag, 'role': 'A', 'ns': proto, 'steps': 10})
    JOBS.append({'frame': frame, 'tag': tag, 'role': 'B50', 'ns': proto, 'steps': 50})
for i, alt in enumerate((1001, 1002, 1003, 1004), start=1):
    JOBS.append({'frame': 440, 'tag': 'f440', 'role': 'C%d' % i, 'ns': alt, 'steps': 10})
JOBS.append({'frame': 440, 'tag': 'f440', 'role': 'C1_50', 'ns': 1001, 'steps': 50})
(OUT / 'predeclared_jobs.json').write_text(json.dumps(
    {**D.MARKERS, 'seed': SEED, 'protocol_steps': PROTOCOL_STEPS, 'jobs': JOBS,
     'hypotheses': {'M1': '10-step Euler discretisation produces the chatter',
                    'M2': 'learned conditional distribution is chattery at these states',
                    'M3': 'only the initial noise draw matters'}}, indent=2))

results = []
for job in JOBS:
    name = 's%d_%s_%s' % (SEED, job['tag'], job['role'])
    rec = run_job(job['frame'], job['ns'], job['steps'], name)
    rec['role'] = job['role']
    if job['role'] == 'A':
        rec['gate'] = gate(job['frame'], rec)
    results.append(rec)
    (OUT / 'summary.json').write_text(json.dumps({**D.MARKERS, 'complete': False,
                                                  'results': results}, indent=2, default=str))

# comparison
by = {(r['frame'], r['role']): r for r in results}
cmp = {}
for frame in (440, 80):
    a = by.get((frame, 'A'))
    b = by.get((frame, 'B50'))
    if a and b:
        cmp['f%d_A_vs_B50' % frame] = {
            'flips_A': a['flips_32'], 'flips_B50': b['flips_32'],
            'ratio': b['flips_32'] / a['flips_32'] if a['flips_32'] else None,
            'sign10_equal': a['sign10'] == b['sign10'],
            'sign10_A': a['sign10'], 'sign10_B50': b['sign10'],
        }
cs = [r for r in results if r['role'].startswith('C') and r['steps'] == 10]
if cs:
    cmp['f440_seed_spread'] = {
        'flips': [r['flips_32'] for r in cs],
        'sign10': [r['sign10'] for r in cs],
        'A_sign10': by[(440, 'A')]['sign10'],
        'per_row_agreement_with_A': [
            float(np.mean([(c['sign32'][t] == by[(440, 'A')]['sign32'][t]) for c in cs]))
            for t in range(32)],
    }
(OUT / 'comparison.json').write_text(json.dumps({**D.MARKERS, 'comparison': cmp}, indent=2, default=str))
(OUT / 'summary.json').write_text(json.dumps({**D.MARKERS, 'complete': True,
                                              'comparison': cmp, 'results': results},
                                             indent=2, default=str))
print('COMPARISON', json.dumps(cmp), flush=True)
print('ALL_DONE', flush=True)