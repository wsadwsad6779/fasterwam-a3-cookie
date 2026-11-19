"""v4 continuation of the segmented action-expert adaptation (20261119).

Same as train_action_expert_segmented_20261119.py except:
- resumes from the v3 step-12000 last.pt (copied into TRAIN_WORK by the driver)
- TOTAL_TARGET defaults to 16000 (cosine over 16k; 12000 was the v3 authorized cap)
- gripper-aware sampling: windows whose label dim-6 contains an OPEN frame or a
  sign transition are replicated GRIP_REPLICATE times in the sampling pool, so the
  97.5%-close majority label cannot collapse the gripper dim to always-close
- milestones 13000/14000/15000/16000 write weights-only deltas to WORK
"""
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get('TRAIN_DATA_DIR', str(ROOT / 'teacher_diverse_20261119')))
WORK = Path(os.environ.get('TRAIN_WORK', '/root/repair_runs/action_expert_v4'))
PUB = Path(os.environ.get('TRAIN_PUB', str(ROOT / 'model_repair_runs' / 'action_expert_v4')))
SEG_STEPS = int(os.environ.get('SEG_STEPS', '1000'))
TOTAL_TARGET = int(os.environ.get('TOTAL_TARGET', '16000'))
LR = float(os.environ.get('TRAIN_LR', '5e-5'))
SEED = int(os.environ.get('TRAIN_SEED', '20261119'))
MAX_MIN = float(os.environ.get('SEG_MAX_MIN', '55'))
GRIP_REPLICATE = int(os.environ.get('GRIP_REPLICATE', '8'))
TARGETS = ['action_expert']
MILESTONES = {13000, 14000, 15000, 16000}
MILESTONE_WEIGHTS = MILESTONES

if not (DATA / 'arrays.npz').exists() or not (DATA / 'manifest.json').exists():
    raise SystemExit('DATA_GUARD: missing %s arrays.npz/manifest.json' % DATA)
arr = np.load(DATA / 'arrays.npz', allow_pickle=True)
ep_start = [int(x) for x in arr['ep_start']]
ep_len = [int(x) for x in arr['ep_len']]
manifest = json.loads((DATA / 'manifest.json').read_text())
splits = [str(e['split']) for e in manifest['episodes']]
if len(splits) != len(ep_len):
    raise SystemExit('DATA_GUARD: manifest episodes %d != arrays %d' % (len(splits), len(ep_len)))
if ep_start != list(np.cumsum([0] + ep_len[:-1])):
    raise SystemExit('DATA_GUARD: ep_start not cumulative')
if float(np.mean(np.all(arr['proprio'] == 0, axis=1))) > 0.05:
    raise SystemExit('DATA_GUARD: proprio corrupt')
if not bool(arr['EXPERT_ACTIONS']):
    raise SystemExit('DATA_GUARD: EXPERT_ACTIONS false')
if 'train' not in splits or 'val' not in splits:
    raise SystemExit('DATA_GUARD: manifest lacks train/val splits')
if splits != sorted(splits, key=lambda s: 0 if s == 'train' else 1):
    raise SystemExit('DATA_GUARD: train episodes must precede val episodes')

WORK.mkdir(parents=True, exist_ok=True)
PUB.mkdir(parents=True, exist_ok=True)
if shutil.disk_usage(str(WORK)).free < 12_000_000_000:
    raise SystemExit('DISK_GUARD: <12GB free at %s' % WORK)
if shutil.disk_usage(str(PUB)).free < 3_000_000_000:
    raise SystemExit('DISK_GUARD: <3GB free at %s' % PUB)

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
random.seed(SEED)

model = instantiate(doc.model, model_dtype=torch.bfloat16, device='cuda')
model.load_checkpoint(str(ROOT.parent / 'step_040000_merged.pt'))
model = model.to('cuda')
model.train()
if getattr(model, 'text_encoder', None) is not None:
    model.text_encoder.eval()
for p in model.parameters():
    p.requires_grad_(False)
trainable = []
for name, p in model.named_parameters():
    if any(t in name for t in TARGETS):
        p.requires_grad_(True)
        trainable.append(name)
params = [p for p in model.parameters() if p.requires_grad]
print('[train] tensors', len(trainable), 'params', sum(p.numel() for p in params), flush=True)

_TEXT_CACHE = {}


def _fresh_text_context(self, prompt):
    key = str(prompt)
    if key not in _TEXT_CACHE:
        with torch.no_grad():
            _TEXT_CACHE[key] = model.encode_prompt(key)
    return _TEXT_CACHE[key]


from fasterwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
RobotVideoDataset._get_cached_text_context = _fresh_text_context
from fasterwam.datasets.lerobot.a3_video_dataset import A3VideoDataset

processor = instantiate(doc.data.train.processor)
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
if sum(n_per_ep) != len(ds):
    raise SystemExit('window accounting mismatch: %d vs %d' % (sum(n_per_ep), len(ds)))
edges = np.concatenate([[0], np.cumsum(n_per_ep)])
train_idx = np.concatenate([np.arange(edges[k], edges[k + 1])
                            for k, s in enumerate(splits) if s == 'train'])
val_pool = np.concatenate([np.arange(edges[k], edges[k + 1])
                           for k, s in enumerate(splits) if s == 'val'])
n_val = min(12, len(val_pool))
val_idx = val_pool[np.linspace(0, len(val_pool) - 1, n_val).astype(int)]
print('[data] windows', len(ds), 'train', len(train_idx), 'val', len(val_idx),
      'train_eps', splits.count('train'), 'val_eps', splits.count('val'), flush=True)

NF = int(doc.data.train.num_frames)
_g = np.asarray(arr['action'])[:, 6]
_grip_windows = []
for _i in train_idx:
    _i = int(_i)
    _k = int(np.searchsorted(edges, _i, side='right') - 1)
    _s = _i - int(edges[_k])
    _seg = _g[int(ep_start[_k]) + _s: int(ep_start[_k]) + _s + NF]
    if len(_seg) and (bool((_seg < 0).any())
                      or bool((np.sign(_seg[1:]) != np.sign(_seg[:-1])).any())):
        _grip_windows.append(_i)
_grip_windows = np.asarray(_grip_windows, dtype=int)
pool = (np.concatenate([train_idx] + [_grip_windows] * GRIP_REPLICATE)
        if len(_grip_windows) else train_idx)
print('[grip] open/transition windows %d/%d replicate %d pool %d'
      % (len(_grip_windows), len(train_idx), GRIP_REPLICATE, len(pool)), flush=True)

data_sha = hashlib.sha256((DATA / 'arrays.npz').read_bytes()).hexdigest()
frames_size = (DATA / 'frames.dat').stat().st_size


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


opt = torch.optim.AdamW(params, lr=LR)


def lr_at(step):
    if step >= TOTAL_TARGET:
        return LR * 0.1
    import math
    return LR * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / TOTAL_TARGET)))


def cpu_opt_state(st):
    out = {}
    for k, v in st.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        else:
            out[k] = v
    return out


def weights_only(path, step):
    sd = model.state_dict()
    torch.save({'delta': {k: sd[k].detach().cpu() for k in trainable},
                'trainable': trainable, 'step': int(step),
                'base': str(ROOT.parent / 'step_040000_merged.pt'),
                'structure': 'action_expert full local fine-tune (not LoRA)',
                'data_sha256': data_sha, 'optimizer_state': 'NOT_IN_WEIGHTS_ONLY'}, path)


last = WORK / 'last.pt'
start_step = 0
if os.environ.get('TRAIN_RESUME', '0') == '1' and last.exists():
    ck = torch.load(last, map_location='cuda', weights_only=False)
    model.load_state_dict({k: v for k, v in ck['delta'].items()}, strict=False)
    opt.load_state_dict(ck['optimizer'])
    torch.set_rng_state(ck['rng']['torch'].cpu())
    np.random.set_state(ck['rng']['np'])
    random.setstate(ck['rng']['py'])
    start_step = int(ck['step'])
    print('[resume] from step', start_step, flush=True)

end_step = start_step + SEG_STEPS
t0 = time.time()
losses = []
for step in range(start_step + 1, end_step + 1):
    for g in opt.param_groups:
        g['lr'] = lr_at(step)
    i = int(pool[int(np.random.randint(len(pool)))])
    sample = to_batch(ds[i])
    loss, parts = model.training_loss(sample)
    if not torch.isfinite(loss):
        raise SystemExit('NONFINITE_LOSS at step %d' % step)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    gnorm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
    opt.step()
    losses.append(float(loss.detach()))
    if step % 10 == 0 or step == start_step + 1:
        msg = 'step %d lr %.2e loss %.4f parts %s gnorm %.3f vram %.2fGB elapsed %.1fs' % (
            step, opt.param_groups[0]['lr'], losses[-1],
            {k: round(v, 4) for k, v in (parts.items() if isinstance(parts, dict) else [])},
            gnorm, torch.cuda.max_memory_allocated() / 1e9, time.time() - t0)
        print(msg, flush=True)
    if (time.time() - t0) > MAX_MIN * 60:
        print('[time-cap] stopping at step', step, flush=True)
        end_step = step
        break
    if step % 1000 == 0:
        torch.save({'delta': {k: model.state_dict()[k].detach().cpu() for k in trainable},
                    'optimizer': cpu_opt_state(opt.state_dict()),
                    'step': step, 'trainable': trainable,
                    'rng': {'torch': torch.get_rng_state(), 'np': np.random.get_state(),
                            'py': random.getstate()},
                    'data_sha256': data_sha, 'lr': opt.param_groups[0]['lr']}, last)
        print('[ckpt] saved last.pt at step', step, flush=True)

val = []
val_parts = []
with torch.no_grad():
    for i in val_idx:
        s = to_batch(ds[i])
        l, pp = model.training_loss(s)
        val.append(float(l))
        val_parts.append({k: float(v) for k, v in pp.items() if isinstance(v, (int, float))}
                         if isinstance(pp, dict) else {})

torch.save({'delta': {k: model.state_dict()[k].detach().cpu() for k in trainable},
            'optimizer': cpu_opt_state(opt.state_dict()),
            'step': end_step, 'trainable': trainable,
            'rng': {'torch': torch.get_rng_state(), 'np': np.random.get_state(),
                    'py': random.getstate()},
            'data_sha256': data_sha, 'lr': opt.param_groups[0]['lr']}, last)
if end_step in MILESTONE_WEIGHTS:
    weights_only(WORK / ('delta_step_%06d.pt' % end_step), end_step)
if end_step in MILESTONES:
    weights_only(WORK / 'delta_latest.pt', end_step)

val_action = [pp.get('loss_action') for pp in val_parts if 'loss_action' in pp]
val_video = [pp.get('loss_video') for pp in val_parts if 'loss_video' in pp]
val_force = [pp.get('loss_force') for pp in val_parts if 'loss_force' in pp]
rec = {'step_range': [start_step + 1, end_step],
       'loss_mean': float(np.mean(losses)) if losses else None,
       'loss_mean_last10': float(np.mean(losses[-10:])) if losses else None,
       'val_loss': float(np.mean(val)), 'val_parts': val_parts,
       'val_action': float(np.mean(val_action)) if val_action else None,
       'val_video': float(np.mean(val_video)) if val_video else None,
       'val_force': float(np.mean(val_force)) if val_force else None,
       'lr_end': opt.param_groups[0]['lr'], 'wall_s': round(time.time() - t0, 1),
       'data_sha256': data_sha, 'frames_bytes': frames_size,
       'val_episodes': [i for i, s in enumerate(splits) if s == 'val'],
       'val_episode_is_holdout': True, 'val_layouts_off_grid': True,
       'grip_windows': int(len(_grip_windows)), 'grip_replicate': GRIP_REPLICATE,
       'structure': 'action_expert full local fine-tune'}
seg_log = WORK / 'segments.jsonl'
with open(seg_log, 'a') as f:
    f.write(json.dumps(rec) + '\n')
print(json.dumps(rec), flush=True)
print('SEGMENT_DONE', flush=True)
