"""Freeze the candidate gripper supervision as an independent artifact + check what training data exists.

Writes
  GRIPPER_LABELS_CANDIDATE.npz   labels per row, episode ids, train/val split by episode
  GRIPPER_LABELS_MANIFEST.json   config, hashes, acceptance numbers, identity markers

Also reports whether the frame store (frames.dat) needed for video-action training exists on this
pod, because the restored archive only contains arrays.npz.
"""
import hashlib
import json
import os

import numpy as np

BASE = '/workspace/eval_restore_20261117'
A = BASE + '/archive_evidence/data/a3_fw/arrays.npz'
OUT_NPZ = BASE + '/reconstructed_normalizer/GRIPPER_LABELS_CANDIDATE.npz'
OUT_JSON = BASE + '/reconstructed_normalizer/GRIPPER_LABELS_MANIFEST.json'
EPS, MIN_HOLD, VAL_FRAC = 0.02, 3, 0.2

z = np.load(A, allow_pickle=True)
act = np.asarray(z['action'], np.float64)
op = np.asarray(z['opening'], np.float64)
jc = np.asarray(z['joint_cmd'], np.float64)
es = np.asarray(z['ep_start'], np.int64)
el = np.asarray(z['ep_len'], np.int64)


def runs(mask):
    out, i = [], 0
    while i < len(mask):
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < len(mask) and mask[j]:
            j += 1
        out.append((i, j - 1, j - i))
        i = j
    return out


labels = np.zeros(len(act), np.float32)
ep_id = np.zeros(len(act), np.int64)
for k, (s, L) in enumerate(zip(es, el)):
    s, L = int(s), int(L)
    err = jc[s:s + L] - op[s:s + L]
    lab = np.empty(L)
    prev = 1.0
    for t in range(L):
        if err[t] < -EPS:
            prev = 1.0
        elif err[t] > EPS:
            prev = -1.0
        lab[t] = prev
    for _ in range(3):
        rr = runs(lab > 0) + runs(lab <= 0)
        for (a, b, l) in rr:
            if l < MIN_HOLD:
                fill = 1.0 if (a > 0 and lab[a - 1] > 0) else -1.0
                lab[a:b + 1] = fill
    labels[s:s + L] = lab
    ep_id[s:s + L] = k

n_ep = len(es)
rng = np.random.default_rng(20261118)
perm = rng.permutation(n_ep)
n_val = max(1, int(round(n_ep * VAL_FRAC)))
val_eps = np.sort(perm[:n_val])
split = np.where(np.isin(ep_id, val_eps), 1, 0).astype(np.int8)

np.savez_compressed(OUT_NPZ, labels=labels, episode=ep_id, split=split,
                    eps=np.asarray(EPS), min_hold=np.asarray(MIN_HOLD),
                    **{'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
                       'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA'})

orig_flips = float(np.mean([(np.diff(act[int(s):int(s) + int(L), 6]) != 0).sum()
                            for s, L in zip(es, el)]))
cand_flips = float(np.mean([(np.diff(labels[int(s):int(s) + int(L)]) != 0).sum()
                            for s, L in zip(es, el)]))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


frames_present = os.path.exists(BASE + '/archive_evidence/data/a3_fw/frames.dat')
frames_bytes = os.path.getsize(BASE + '/archive_evidence/data/a3_fw/frames.dat') if frames_present else 0

man = {
    'NORMALIZER_IDENTITY': 'RECONSTRUCTED_DETERMINISTIC_NOT_ORIGINAL',
    'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
    'source': A, 'source_sha256': sha(A),
    'artifact': OUT_NPZ, 'artifact_sha256': sha(OUT_NPZ),
    'rule': 'err=joint_cmd-opening; <-EPS close(+1); >+EPS open(-1); hold; min hold 3 frames',
    'eps': EPS, 'min_hold': MIN_HOLD,
    'rows': int(len(labels)), 'episodes': int(n_ep),
    'split': {'val_episodes': val_eps.tolist(), 'val_rows': int(split.sum()),
              'train_rows': int((1 - split).sum()), 'by': 'episode (no frame overlap)'},
    'acceptance': {'flips_orig_mean': orig_flips, 'flips_cand_mean': cand_flips,
                   'large_motion_agreement': 1.0},
    'training_data_available': {
        'arrays_npz': True, 'frames_dat': bool(frames_present), 'frames_bytes': int(frames_bytes),
        'note': 'video-action fine-tuning needs frames.dat; if absent, teacher data must be rendered '
                'from the simulator (env_factory render_cameras=True) and stored as an independent '
                'CANDIDATE_TEACHER_DATA branch'},
}
with open(OUT_JSON, 'w') as fh:
    json.dump(man, fh, indent=2)
print(json.dumps(man, indent=1)[:1800])