cd /workspace
grep -nE 'proprio' /workspace/FasterWAM/src/fasterwam/models/wan22/fastwam.py | grep -inE 'save|load|state|dict|checkpoint' | head -20
grep -nE 'def save_light_state|def load_light_state|proprio|trainable' /workspace/_r4_stage/up/light_state.new.py | head -30
python3 - <<'PY'
import torch
sd = torch.load('/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/adapter.pt', map_location='cpu', weights_only=False)
print(type(sd), list(sd.keys())[:6] if isinstance(sd, dict) else sd)
if isinstance(sd, dict):
    inner = sd.get('adapter') if 'adapter' in sd else sd
    if isinstance(inner, dict):
        ks = [k for k in inner if 'proprio' in k]
        print('proprio keys', ks)
        for k in ks:
            print(' ', k, tuple(inner[k].shape))
        print('total tensors', len(inner), 'lora tensors', sum(1 for k in inner if 'lora' in k))
PY
