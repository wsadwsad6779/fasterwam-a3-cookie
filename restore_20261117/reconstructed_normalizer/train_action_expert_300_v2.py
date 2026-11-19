"""Bounded action-expert fine-tune v2 on the corrected teacher data (300 steps, fixed budget).

Fixes versus train_lora_300.py:
  * data = teacher_replay_20261119 (ep_start = [0,600,1200,1800]; proprio = deployment layout)
  * DATA_GUARD checks the corrected offsets and a non-degenerate proprio
  * episode-level split: seeds 41/42/43 train, seed 44 validation (no window leakage)
  * checkpoints contain ONLY the 602 trainable action-expert tensors (~1.13 GB), never the
    24 GB full state_dict; a disk preflight refuses to start when space is insufficient
  * optimizer state is not persisted; there is no pretence of a seamless resume
"""
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
RUN = ROOT / os.environ.get('TRAIN_OUT', 'model_repair_runs/action_expert_300_v2')
DATA = Path(os.environ.get('TRAIN_DATA_DIR', str(ROOT / 'teacher_replay_20261119')))
STEPS = int(os.environ.get('TRAIN_STEPS', '300'))
MAX_MIN = float(os.environ.get('TRAIN_MAX_MIN', '90'))
LR = float(os.environ.get('TRAIN_LR', '1e-4'))
SEED = int(os.environ.get('TRAIN_SEED', '20261118'))
TARGETS = ['action_expert']
VAL_EPISODES = int(os.environ.get('TRAIN_VAL_EPISODES', '1'))

if not (DATA / 'frames.dat').exists() or not (DATA / 'arrays.npz').exists():
    raise SystemExit('DATA_GUARD: %s lacks frames.dat/arrays.npz' % DATA)
arr = np.load(DATA / 'arrays.npz', allow_pickle=True)
ep_start = [int(x) for x in arr['ep_start']]
ep_len = [int(x) for x in arr['ep_len']]
if ep_start != list(np.cumsum([0] + ep_len[:-1])):
    raise SystemExit('DATA_GUARD: ep_start is not cumulative: %s' % ep_start)
if float(np.mean(np.all(arr['proprio'] == 0, axis=1))) > 0.05:
    raise SystemExit('DATA_GUARD: proprio is mostly zero rows (alignment bug not fixed)')
if 'EXPERT_ACTIONS' not in arr.files or not bool(arr['EXPERT_ACTIONS']):
    raise SystemExit('DATA_GUARD: EXPERT_ACTIONS marker missing/false')

RUN.mkdir(parents=True, exist_ok=False)
free = shutil.disk_usage(str(RUN.parent)).free
if free < 5_000_000_000:
    raise SystemExit('DISK_GUARD: only %.2f GB free; a delta snapshot needs ~1.2 GB x3'
                     % (free / 1e9))

E, factory, W, fw = D.setup_protocol()
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

doc = OmegaConf.load(str(ROOT.parent / 'step_040000' / 'run_config.yaml'))
OmegaConf.set_struct(doc, False)
if 'force' in doc.model and 'norm_stats' in doc.model.force:
    doc.model.force.norm_stats.file = str(ROOT.parent / 'step_040000' / 'force_calib_v1.json')
doc.data.train.pretrained_norm_stats = str(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json')
doc.model.load_text_encoder = True
doc.data.train.dataset_dirs = [str(DATA)]
doc.data.train.val_set_proportion = 0.0
OmegaConf.set_struct(doc, True)

torch.manual_seed(SEED)
np.random.seed(SEED)

model = instantiate(doc.model, model_dtype=torch.bfloat16, device='cuda')
model.load_checkpoint(str(ROOT.parent / 'step_040000_merged.pt'))
model = model.to('cuda')
model.train()
if hasattr(model, 'text_encoder') and model.text_encoder is not None:
    model.text_encoder.eval()
for p in model.parameters():
    p.requires_grad_(False)
trainable = []
for name, p in model.named_parameters():
    if any(t in name for t in TARGETS):
        p.requires_grad_(True)
        trainable.append(name)
if not trainable:
    raise SystemExit('no trainable parameters matched %s' % TARGETS)
n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('[train] trainable tensors', len(trainable), 'params', n_train_params, flush=True)

_TEXT_CACHE = {}


def _fresh_text_context(self, prompt):
    key = str(prompt)
    if key not in _TEXT_CACHE:
        with torch.no_grad():
            emb, mask = model.encode_prompt(key)
        _TEXT_CACHE[key] = (emb, mask)
    return _TEXT_CACHE[key]


from fasterwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
RobotVideoDataset._get_cached_text_context = _fresh_text_context

processor = instantiate(doc.data.train.processor)
from fasterwam.datasets.lerobot.a3_video_dataset import A3VideoDataset
ds = A3VideoDataset(dataset_dirs=[str(DATA)], shape_meta=dict(doc.data.train.shape_meta),
                    num_frames=int(doc.data.train.num_frames),
                    video_size=list(doc.data.train.video_size),
                    processor=processor,
                    text_embedding_cache_dir=doc.data.train.get('text_embedding_cache_dir'),
                    context_len=int(doc.data.train.context_len),
                    pretrained_norm_stats=str(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json'),
                    is_training_set=True,
                    force_arrays_path=str(DATA / 'force_arrays.npz'),
                    force_channels=[int(c) for c in doc.data.train.force_channels],
                    force_hist_len=int(doc.data.train.force_hist_len),
                    force_horizon=int(doc.data.train.force_horizon))
n_per_ep = [int(L) - int(doc.data.train.num_frames) + 1 for L in ep_len]
train_windows = int(sum(n_per_ep[:-VAL_EPISODES]))
val_windows = int(sum(n_per_ep[-VAL_EPISODES:]))
if train_windows + val_windows != len(ds):
    raise SystemExit('window accounting mismatch: %d+%d vs %d'
                     % (train_windows, val_windows, len(ds)))
train_idx = list(range(train_windows))
val_idx = list(range(train_windows, len(ds)))[:: max(1, val_windows // 8)][:8]
print('[data] windows', len(ds), 'train', len(train_idx), 'val', len(val_idx), flush=True)


def to_batch(sample):
    out = {}
    for k, v in sample.items():
        if torch.is_tensor(v):
            if k == 'context' and v.ndim == 3:
                out[k] = v.to('cuda')
            elif k == 'context_mask' and v.ndim == 2:
                out[k] = v.to('cuda')
            else:
                out[k] = v.unsqueeze(0).to('cuda')
        else:
            out[k] = v
    return out


opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)


def save_delta(path, step):
    sd = model.state_dict()
    torch.save({'delta': {k: sd[k].detach().to('cpu') for k in trainable},
                'trainable': trainable, 'step': int(step),
                'base': str(ROOT.parent / 'step_040000_merged.pt'),
                'optimizer_state': 'NOT_SAVED'}, path)
    return path.stat().st_size


def check_space(need):
    free = shutil.disk_usage(str(RUN.parent)).free
    if free < need:
        raise SystemExit('DISK_GUARD: %.2f GB free < %.2f GB needed'
                         % (free / 1e9, need / 1e9))


log = open(RUN / 'train.log', 'w')
t0 = time.time()
losses = []
for step in range(1, STEPS + 1):
    i = train_idx[int(np.random.randint(len(train_idx)))]
    sample = ds[i]
    sample = to_batch(sample)
    loss, parts = model.training_loss(sample)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0))
    opt.step()
    losses.append(float(loss.detach()))
    if step % 10 == 0 or step == 1:
        msg = 'step %d loss %.4f parts %s gnorm %.3f vram %.2fGB elapsed %.1fs' % (
            step, losses[-1], {k: round(v, 4) for k, v in
                               (parts.items() if isinstance(parts, dict) else [])},
            gnorm, torch.cuda.max_memory_allocated() / 1e9, time.time() - t0)
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()
    if (time.time() - t0) > MAX_MIN * 60:
        log.write('time cap reached at step %d\n' % step)
        break
    if step % 100 == 0:
        check_space(3_000_000_000)
        size = save_delta(RUN / ('delta_step_%06d.pt' % step), step)
        print('[ckpt] step %d delta %.2f GB' % (step, size / 1e9), flush=True)

val = []
with torch.no_grad():
    for i in val_idx:
        s = to_batch(ds[i])
        l, _ = model.training_loss(s)
        val.append(float(l))
check_space(2_000_000_000)
ckpt = RUN / 'delta_final.pt'
size = save_delta(ckpt, len(losses))
man = {**D.MARKERS, 'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
       'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA',
       'base_checkpoint': str(ROOT.parent / 'step_040000_merged.pt'),
       'data_dir': str(DATA), 'steps': len(losses), 'lr': LR, 'seed': SEED,
       'trainable_tensors': len(trainable), 'trainable_params': int(n_train_params),
       'structure': 'action_expert full local fine-tune (no LoRA factors)',
       'split': {'train_episodes': ep_len[:-VAL_EPISODES],
                 'val_episodes': ep_len[-VAL_EPISODES:],
                 'train_windows': len(train_idx), 'val_windows': len(val_idx),
                 'test_seeds': [29, 30], 'test_never_trained': True},
       'loss_first10': losses[:10], 'loss_mean_last10': float(np.mean(losses[-10:])),
       'val_loss_episode_holdout': float(np.mean(val)) if val else None,
       'checkpoint': str(ckpt), 'checkpoint_bytes': size,
       'optimizer_state': 'NOT_SAVED (no seamless resume)',
       'cursor': 'model input = images/proprio/force/text only; no oracle fields',
       'wall_s': round(time.time() - t0, 1)}
(RUN / 'manifest.json').write_text(json.dumps(man, indent=2, default=str))
print(json.dumps({k: man[k] for k in ('steps', 'loss_mean_last10', 'val_loss_episode_holdout',
                                      'checkpoint_bytes')}), flush=True)
print('TRAIN_DONE', flush=True)
