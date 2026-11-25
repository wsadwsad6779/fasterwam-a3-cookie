import importlib.util
import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as Fn

sys.path.insert(0, '/workspace')
from t7_data import A3BatchSource  # noqa: E402

CKPT = '/workspace/fwv1/train_runs/t7_21k_v3run/step_015000.pt'
P = '/workspace/fwv1/p_items'
spec = importlib.util.spec_from_file_location('t7ev', '/workspace/_t7_eval.py')
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

model, film, state = ev.build_model(CKPT)
model.train()
src = A3BatchSource(split='val', batch_size=2, seed=20261125, balanced=False)
bb = src.next_batch()
sample = {'video': bb['video'].cuda(), 'context': bb['context'].cuda(),
          'context_mask': bb['context_mask'].cuda(), 'action': bb['action'].cuda(),
          'proprio': bb['proprio'].cuda(),
          'action_is_pad': bb['action_is_pad'].cuda(),
          'image_is_pad': bb['image_is_pad'].cuda()}
cap = []
orig = Fn.mse_loss


def patched(pred, target, *a, **k):
    if torch.is_tensor(pred) and pred.dim() == 3 and pred.shape[-1] == 7:
        cap.append((pred, target))
    return orig(pred, target, *a, **k)


Fn.mse_loss = patched
model.zero_grad()
loss, ld = model.training_loss(sample)
Fn.mse_loss = orig
if not cap:
    raise SystemExit('NO_ACTION_LOSS_CAPTURED')
pred, target = cap[-1]
with torch.no_grad():
    per_dim = ((pred - target) ** 2).mean(dim=(0, 1)).tolist()
    p6 = pred[..., 6].detach().float().cpu().numpy().reshape(-1)
    t6 = target[..., 6].detach().float().cpu().numpy().reshape(-1)
dim_grads = {}
for d in range(7):
    model.zero_grad()
    ld_ = ((pred[..., d] - target[..., d]) ** 2).mean()
    ld_.backward(retain_graph=(d < 6))
    dim_grads[d] = math.sqrt(sum(float(p.grad.norm()) ** 2
                                  for p in model.action_expert.parameters()
                                  if p.grad is not None))
grads = []
for name, p in model.action_expert.named_parameters():
    if p.grad is not None:
        grads.append((name, float(p.grad.norm()), list(p.shape)))
tot = dim_grads[6]
ratio = dim_grads[6] / (sum(dim_grads[d] for d in range(6)) / 6.0) if any(dim_grads.values()) else None
top = sorted(grads, key=lambda x: -x[1])[:10]
stats6 = {'target_lt_0.30': float((t6 < 0.30).mean()),
          'target_open_gt_0.8': float((t6 >= 0.8).mean()),
          'target_mid_0.30_0.80': float(((t6 >= 0.30) & (t6 < 0.8)).mean()),
          'target_median': float(np.median(t6)),
          'pred_median': float(np.median(p6)),
          'pred_lt_0.30': float((p6 < 0.30).mean()),
          'pred_open_gt_0.8': float((p6 >= 0.8).mean())}
out = {'ts': __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ', __import__('time').gmtime()),
       'artifact': 'DIM7_LOSS_GRAD_AUDIT',
       'checkpoint': CKPT,
       'checkpoint_sha256': '83aacf23cc146af4b33027ab1cfaea0ce8fd0a0aac6c28cf683fdfc69dbab2e1',
       'batch': {'split': 'val', 'batch_size': 2, 'seed': 20261125, 'episode': bb['meta']['episode']},
       'per_dim_loss': per_dim,
       'dim6_stats': stats6,
       'dim_grad_total_norm': {str(k): v for k, v in dim_grads.items()},
       'ratio_dim6_to_other6': ratio,
       'loss_action_total': float(ld.get('loss_action')) if ld.get('loss_action') is not None else None,
       'grad_total_norm_action_expert': tot,
       'grad_top': [{'name': n, 'norm': g, 'shape': s} for n, g, s in top],
       'n_params_with_grad': len(grads),
       'read_only': True,
       'note': 'one forward/backward on a val batch; no optimizer step, no weight change'}
json.dump(out, open(P + '/DIM7_LOSS_GRAD_AUDIT.json', 'w'), indent=1)
print(json.dumps({'per_dim_loss': per_dim, 'grad_total': tot, 'n': len(grads)}))