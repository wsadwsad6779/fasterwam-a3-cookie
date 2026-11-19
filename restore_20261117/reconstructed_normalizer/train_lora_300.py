"""Bounded model-repair training driver (300 optimizer steps, independent checkpoint).

Hard guards (refuses to run instead of training on the wrong targets):
  * dataset dir must exist and contain frames.dat + arrays.npz
  * arrays.npz must carry expert actions (N,7) and the marker EXPERT_ACTIONS=True
    (teacher episodes recorded from the policy are NOT acceptable action targets)
  * original 40k merged checkpoint is only read, never written

Adaptation: all parameters frozen, then a bounded whitelist of modules unfrozen (LoRA-style
low-rank adaptation is used when available; the exact module list is written to the manifest).
Stops at TRAIN_STEPS (default 300) or earlier when TRAIN_MAX_MIN minutes elapse.
"""
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
RUN = ROOT / os.environ.get('TRAIN_OUT', 'model_repair_runs/lora_300')
DATA = Path(os.environ.get('TRAIN_DATA_DIR', str(ROOT / 'teacher_replay_20261118')))
STEPS = int(os.environ.get('TRAIN_STEPS', '300'))
SMOKE = os.environ.get('TRAIN_SMOKE', '0') == '1'
MAX_MIN = float(os.environ.get('TRAIN_MAX_MIN', '90'))
LR = float(os.environ.get('TRAIN_LR', '1e-4'))
SEED = int(os.environ.get('TRAIN_SEED', '20261118'))
TARGETS = os.environ.get('TRAIN_TARGETS', 'action_expert').split(',')

if not (DATA / 'frames.dat').exists() or not (DATA / 'arrays.npz').exists():
    raise SystemExit('DATA_GUARD: %s lacks frames.dat/arrays.npz; refusing to train. '
                     'Teacher-policy episodes are not valid action targets.' % DATA)
if not (DATA / 'force_arrays.npz').exists():
    raise SystemExit('DATA_GUARD: %s lacks force_arrays.npz' % DATA)
arr = np.load(DATA / 'arrays.npz', allow_pickle=True)
if 'action' not in arr.files or arr['action'].shape[1] != 7:
    raise SystemExit('DATA_GUARD: arrays.npz has no (N,7) action')
if 'EXPERT_ACTIONS' in arr.files and not bool(arr['EXPERT_ACTIONS']):
    raise SystemExit('DATA_GUARD: EXPERT_ACTIONS marker is false')

RUN.mkdir(parents=True, exist_ok=False)

E, factory, W, fw = D.setup_protocol()
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fasterwam.datasets.lerobot.a3_video_dataset import A3VideoDataset
from fasterwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor

doc = OmegaConf.load(str(ROOT.parent / 'step_040000' / 'run_config.yaml'))
OmegaConf.set_struct(doc, False)
if 'force' in doc.model and 'norm_stats' in doc.model.force:
    doc.model.force.norm_stats.file = str(ROOT.parent / 'step_040000' / 'force_calib_v1.json')
doc.data.train.pretrained_norm_stats = str(ROOT / 'dataset_stats_14d_RECONSTRUCTED.json')
doc.model.load_text_encoder = True   # training-time text cache was lost; encode fresh with the original T5
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
    for p in model.text_encoder.parameters():
        p.requires_grad_(False)
for p in model.parameters():
    p.requires_grad_(False)
trainable = []
for name, p in model.named_parameters():
    if any(t.strip() and t.strip() in name for t in TARGETS):
        p.requires_grad_(True)
        trainable.append(name)
if not trainable:
    raise SystemExit('no trainable parameters matched %s' % TARGETS)
print('[train] trainable tensors', len(trainable), 'params',
      sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)

# text cache: the original training cache was lost with the old container; encode the prompt
# with the original T5 through the loaded model and cache it in-process (marked as regenerated).
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
print('[data] windows', len(ds), flush=True)



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

idx_val = list(range(max(0, len(ds) - 8), len(ds)))
log = open(RUN / 'train.log', 'w')
t0 = time.time()
losses = []
for step in range(1, STEPS + 1):
    i = int(np.random.randint(len(ds)))
    sample = ds[i]
    sample = to_batch(sample)
    try:
        loss, parts = model.training_loss(sample)
    except Exception as exc:
        log.write('step %d FAILED %r\n' % (step, exc)); log.flush()
        raise
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
        print(msg, flush=True); log.write(msg + '\n'); log.flush()
    if SMOKE and step >= 2:
        break
    if (time.time() - t0) > MAX_MIN * 60:
        log.write('time cap reached at step %d\n' % step); break
    if step % 100 == 0:
        torch.save({'model': model.state_dict(), 'step': step, 'trainable': trainable},
                   RUN / ('step_%06d.pt' % step))

val = []
with torch.no_grad():
    for i in idx_val[:5]:
        s = ds[i]
        s = to_batch(s)
        l, _ = model.training_loss(s)
        val.append(float(l))
ckpt = RUN / 'final.pt'
torch.save({'model': model.state_dict(), 'step': len(losses), 'trainable': trainable,
            'base': str(ROOT.parent / 'step_040000_merged.pt')}, ckpt)
man = {**D.MARKERS, 'base_checkpoint': str(ROOT.parent / 'step_040000_merged.pt'),
       'data_dir': str(DATA), 'steps': len(losses), 'lr': LR, 'seed': SEED,
       'trainable_tensors': len(trainable), 'trainable_names_head': trainable[:12],
       'loss_first10': losses[:10], 'loss_mean_last10': float(np.mean(losses[-10:])),
       'val_loss': float(np.mean(val)) if val else None,
       'checkpoint': str(ckpt), 'smoke': SMOKE,
       'cursor': 'model input = images/proprio/force/text only; no oracle fields',
       'wall_s': round(time.time() - t0, 1)}
(RUN / 'manifest.json').write_text(json.dumps(man, indent=2, default=str))
print(json.dumps({k: man[k] for k in ('steps', 'loss_mean_last10', 'val_loss', 'wall_s',
                                      'trainable_tensors')}), flush=True)
print('TRAIN_DONE', flush=True)