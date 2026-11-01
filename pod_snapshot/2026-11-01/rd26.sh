cd /workspace
grep -nE 'def |format|lora|proprio|state_dict' /workspace/FasterWAM/src/fasterwam/light_state.py | head -40
echo '=== r3 merged has proprio? ==='
python3 - <<'PY'
import torch
p = '/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt'
obj = torch.load(p, map_location='cpu', weights_only=False)
print(type(obj), list(obj.keys())[:10] if isinstance(obj, dict) else '')
if isinstance(obj, dict):
    for k in obj:
        if 'proprio' in str(k):
            print('top-level proprio', k)
    inner = obj.get('model') or obj.get('state_dict') or obj
    if isinstance(inner, dict):
        ks = [k for k in inner if 'proprio' in str(k)]
        print('inner proprio keys', ks[:5])
        for k in ks:
            try: print(' ', k, tuple(inner[k].shape))
            except Exception: pass
PY
echo '=== gate status ==='
ls /workspace/a3_eval/ | grep replay
tail -c 200 /workspace/gate.out
tail -c 200 /workspace/gate10.out
pgrep -af replay_forces | head -3
