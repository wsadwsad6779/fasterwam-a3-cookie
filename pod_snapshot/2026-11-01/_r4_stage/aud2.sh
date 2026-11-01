#!/bin/bash
R4=/workspace/a3_train_runs/a3_lora_r4
R3=/workspace/a3_train_runs/a3_lora_r3
echo "== r4 config.yaml =="
cat $R4/config.yaml
echo "== trainer_state step_004000 =="
cat $R4/checkpoints/light/step_004000/trainer_state.json
echo "== state dir =="
ls -la $R4/checkpoints/state | head -8
echo "== weights dir =="
ls -la $R4/checkpoints/weights | head -8
echo "== r3 config diff (r3 vs r4) =="
diff $R3/config.yaml $R4/config.yaml
echo "== r3 ckpt tree =="
ls -la $R3/checkpoints | head -10
echo "== preflight =="
cat $R4/preflight.txt
echo "== train.log tail =="
tail -22 $R4/train.log
echo "== train.log resume lines =="
grep -in "resume|global_step|start_step|sampler|rng|loading|loaded" $R4/train.log | head -20
echo "== DONE =="