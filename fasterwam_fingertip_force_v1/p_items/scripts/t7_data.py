"""On-the-fly 845-episode training batch provider.

Batches are built from B consecutive windows of one episode so each video
shard is decoded sequentially (AV1 random seeks are the bottleneck).
Window: 9 video frames at stride 4 over 32 action steps; proprio 33 rows;
force history 8 rows ending at the window start (left-padded zeros + mask).
"""
import json
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

VIDEOS = '/workspace/fwv1/data/hf_videos/videos'
EPMETA = '/workspace/fwv1/data/hf_meta/meta/episodes/chunk-000/file-000.parquet'
PARQ = ['/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-000.parquet',
        '/workspace/fwv1/p_items/hf_multi/data/chunk-000/file-001.parquet']
LABELS = '/workspace/fwv1/p_items/A3_845_7D_LABELS_V2.npz'
SPLIT = '/workspace/fwv1/p_items/DATASET_800_SPLIT_MANIFEST.json'
STATS = '/workspace/fwv1/data/a3_fw_845/dataset_stats.json'
FSTATS = '/workspace/fwv1/p_items/FINGERTIP_TRAIN_NORM_STATS.json'
CTX = None
PROMPT = "A video recorded from a robot's point of view executing the following instruction: transfer 10 cookies into target box"

NUM_FRAMES = 33
STRIDE = 4
T_VID = 9
AH = 32
HIST = 8


def _norm_stats():
    st = json.load(open(STATS))
    a = st['action']['default']
    s = st['state']['default']
    return (np.asarray(a['global_min'], np.float32), np.asarray(a['global_max'], np.float32),
            np.asarray(s['global_min'], np.float32), np.asarray(s['global_max'], np.float32))


def _lin(x, lo, hi):
    rng = hi - lo
    rng = np.where(rng < 1e-4, 2.0, rng)
    scale = 2.0 / rng
    off = -1.0 - scale * lo
    return np.clip(x * scale + off, -5.0, 5.0)


class A3BatchSource:
    def __init__(self, split='train', batch_size=8, seed=0):
        self.batch_size = int(batch_size)
        self.rng = random.Random(seed)
        sp = json.load(open(SPLIT))
        self.episodes = [int(x) for x in sp['train_episodes' if split == 'train' else 'val_episodes']]
        meta = pq.read_table(EPMETA).to_pylist()
        self.meta = {int(r['episode_index']): r for r in meta}
        z = np.load(LABELS)
        self.L = z['label'].astype(np.float32)
        self.lab_start = z['ep_start_label'].astype(np.int64)
        self.el = z['ep_len'].astype(np.int64)
        t = pq.read_table(PARQ, columns=['episode_index', 'observation.state', 'observation.force'])
        E = np.asarray(t['episode_index'])
        S = np.asarray(t['observation.state'].to_pylist(), np.float64).astype(np.float32)
        fc = t['observation.force'].combine_chunks()
        F18 = fc.flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(-1, 18).astype(np.float32)
        order = np.argsort(E, kind='stable')
        E = E[order]
        S = S[order]
        F18 = F18[order]
        b = np.flatnonzero(np.diff(E)) + 1
        st = np.r_[0, b]
        self.s_map = {int(E[st[k]]): int(st[k]) for k in range(len(st))}
        self.S = S
        self.F18 = F18
        self.amin, self.amax, self.smin, self.smax = _norm_stats()
        fs = json.load(open(FSTATS))
        self.fmean = np.asarray(fs['mean'], np.float32)
        self.fstd = np.maximum(np.asarray(fs['std'], np.float32), 1e-3)
        self.factive = (np.asarray(fs['std']) > 1e-6).astype(np.float32)
        ctx = torch.load(os.path.join('/workspace/fwv1/data/text_embeds_cache',
                                      _ctx_name()), map_location='cpu')
        self.context = ctx['context'].float()
        self.context_mask = ctx['mask'].bool()
        self._readers = {}
        self._clock = 0

    def _reader(self, path):
        import imageio
        rd = self._readers.get(path)
        if rd is None:
            rd = imageio.get_reader(path)
            self._readers[path] = rd
        return rd

    def next_batch(self):
        B = self.batch_size
        while True:
            ep = self.episodes[self.rng.randrange(len(self.episodes))]
            n = int(self.el[ep])
            if n < B + AH + 2:
                continue
            start = self.rng.randint(0, n - B - AH)
            break
        m = self.meta[ep]
        f_rel = 'observation.images.front/chunk-%03d/file-%03d.mp4' % (
            int(m['videos/observation.images.front/chunk_index']),
            int(m['videos/observation.images.front/file_index']))
        w_rel = 'observation.images.left_wrist/chunk-%03d/file-%03d.mp4' % (
            int(m['videos/observation.images.left_wrist/chunk_index']),
            int(m['videos/observation.images.left_wrist/file_index']))
        f0 = int(round(float(m['videos/observation.images.front/from_timestamp']) * 20))
        w0 = int(round(float(m['videos/observation.images.left_wrist/from_timestamp']) * 20))
        need = B + AH + 1
        rf = self._reader(os.path.join(VIDEOS, f_rel))
        rw = self._reader(os.path.join(VIDEOS, w_rel))
        ff = np.stack([rf.get_data(f0 + start + j) for j in range(need)], 0)
        fw = np.stack([rw.get_data(w0 + start + j) for j in range(need)], 0)
        vids = []
        for i in range(B):
            idx = [i + STRIDE * k for k in range(T_VID)]
            a = torch.tensor(ff[idx]).permute(0, 3, 1, 2).float() / 255.0
            b = torch.tensor(fw[idx]).permute(0, 3, 1, 2).float() / 255.0
            a = F.interpolate(a, size=(224, 224), mode='bilinear', align_corners=False)
            b = F.interpolate(b, size=(224, 224), mode='bilinear', align_corners=False)
            v = torch.cat([a, b], dim=3)
            vids.append(((v - 0.5) / 0.5).permute(1, 0, 2, 3))
        video = torch.stack(vids, 0)
        acts = []
        props = []
        fhist = []
        fmask = []
        s = self.s_map[ep]
        ls = int(self.lab_start[ep])
        for i in range(B):
            acts.append(torch.tensor(_lin(self.L[ls + start + i:ls + start + i + AH],
                                          self.amin, self.amax)))
            props.append(torch.tensor(_lin(self.S[s + start + i:s + start + i + NUM_FRAMES],
                                           self.smin, self.smax)))
            lo = start + i - HIST
            pad = max(0, -lo)
            rows = self.F18[s + max(0, lo):s + start + i, 12:16]
            norm = np.zeros((HIST, 4), np.float32)
            mask = np.zeros(HIST, bool)
            if len(rows):
                v = np.clip((rows - self.fmean) / self.fstd * self.factive, -5, 5)
                norm[HIST - len(rows):] = v
                mask[HIST - len(rows):] = True
            fhist.append(torch.tensor(norm))
            fmask.append(torch.tensor(mask))
        return {
            'video': video,
            'action': torch.stack(acts, 0),
            'proprio': torch.stack(props, 0),
            'force_history': torch.stack(fhist, 0),
            'force_history_mask': torch.stack(fmask, 0),
            'context': self.context.unsqueeze(0).repeat(B, 1, 1),
            'context_mask': self.context_mask.unsqueeze(0).repeat(B, 1),
            'action_is_pad': torch.zeros(B, AH, dtype=torch.bool),
            'image_is_pad': torch.zeros(B, T_VID, dtype=torch.bool),
            'meta': {'episode': ep, 'start': start},
        }


def _ctx_name():
    import hashlib
    h = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
    return '%s.t5_len128.wan22ti2v5b.pt' % h