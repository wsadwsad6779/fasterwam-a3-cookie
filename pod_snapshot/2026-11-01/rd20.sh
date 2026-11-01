cd /workspace
echo '=== a3_2cam proprio ==='
grep -nE 'proprio|action_output_dim|video' /workspace/FasterWAM/configs/data/a3_2cam.yaml | head -20
echo '=== fasterwam.yaml ==='
grep -nE 'proprio_dim|text_dim|action' /workspace/FasterWAM/configs/model/fasterwam.yaml | head -20
echo '=== fastwam proprio_encoder ==='
grep -nE 'proprio_encoder|proprio_dim|_append_proprio' /workspace/FasterWAM/src/fasterwam/models/wan22/fastwam.py | head -20
echo '=== trainer ==='
grep -nE 'proprio_encoder|warmup_steps' /workspace/FasterWAM/src/fasterwam/train/trainer.py | head -20
echo '=== env force layout ==='
sed -n 370,392p /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py
echo '=== arrays keys ==='
python3 - <<'PY'
import numpy as np
z = np.load('/workspace/a3_data/a3_fw/arrays.npz')
print([k for k in z.keys()])
for k in z.keys():
    try: print(k, z[k].shape, z[k].dtype)
    except Exception as e: print(k, 'err')
PY
ls /workspace/a3_data/a3_fw/ | head
tail -c 500 /workspace/gate10.out
