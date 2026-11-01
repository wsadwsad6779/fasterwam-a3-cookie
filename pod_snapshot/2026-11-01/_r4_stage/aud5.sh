#!/bin/bash
echo "== mech_b0 out =="
cat /workspace/_r4_stage/mech_b0.out
echo "== r1 r2 configs =="
ls -la /workspace/a3_train_runs/a3_lora_r1 /workspace/a3_train_runs/a3_lora_r2 2>&1 | head -30
grep -n max_steps /workspace/a3_train_runs/a3_lora_r2/config.yaml 2>&1
echo "== loss series r4 =="
grep -o "step=[0-9]*" /workspace/a3_train_runs/a3_lora_r4/train.log | head -3
tr -d "\r" < /workspace/a3_train_runs/a3_lora_r4/train.log | grep -o "step=[0-9]* .*" | head -2
echo "== scheduler.pt keys =="
source /workspace/activate_fasterwam_a3.sh >/dev/null 2>&1
python3 -c "import torch,json;s=torch.load('/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/scheduler.pt',map_location='cpu',weights_only=False);print(json.dumps({k:(v if not hasattr(v,'shape') else str(v.shape)) for k,v in s.items()},default=str)[:1200])"
echo "== optimizer.pt keys =="
python3 -c "import torch;o=torch.load('/workspace/a3_train_runs/a3_lora_r4/checkpoints/light/step_004000/optimizer.pt',map_location='cpu',weights_only=False);print(type(o),list(o.keys()));print(o['param_groups'][0]);print(len(o['state']),'param states');import itertools;k=next(iter(o['state']));print({kk:str(vv.shape if hasattr(vv,'shape') else vv)[:40] for kk,vv in o['state'][k].items()})"
echo "== DONE =="