cd /workspace
python3 - <<'PY'
import torch
sd = torch.load('/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt', map_location='cpu', weights_only=False)
print('top keys', list(sd.keys()))
print('format', sd['format'])
print('meta keys', list(sd['meta'].keys()) if isinstance(sd['meta'], dict) else sd['meta'])
lo = sd['lora']
print('lora container type', type(lo), 'len', len(lo))
ks = list(lo.keys()) if hasattr(lo, 'keys') else None
print('first keys', ks[:5])
print('proprio inside lora:', [k for k in (ks or []) if 'proprio' in k])
n_lora = sum(1 for k in ks if 'lora' in k.lower())
print('tensors', len(ks), 'lora-named', n_lora)
import numpy as np
tot = sum(int(np.prod(lo[k].shape)) for k in ks)
print('total elements', tot)
opt = torch.load('/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/optimizer.pt', map_location='cpu', weights_only=False)
print('opt type', type(opt))
if isinstance(opt, dict):
    print('opt keys', list(opt.keys())[:8])
    pg = opt.get('param_groups')
    if pg: print('param_groups', len(pg), 'sizes', [len(g['params']) for g in pg])
PY
grep -rnE 'proprio' /workspace/FasterWAM/src/fasterwam/light_state.py 2>/dev/null | head -20
find /workspace/FasterWAM/src/fasterwam -name 'light_state*.py' | head
