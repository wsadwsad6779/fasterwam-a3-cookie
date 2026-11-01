#!/bin/bash
echo "== r1/r2/r3 max_steps =="
for d in r1 r2 r3 r4; do echo -n "$d: "; grep -H "max_steps|learning_rate|output_dir|resume" /workspace/a3_train_runs/a3_lora_$d/config.yaml 2>/dev/null | tr "\n" " "; echo; done
echo "== mask job =="
tail -3 /workspace/_r4_stage/tko_b0.log
echo "== train_light.py =="
cat /workspace/FasterWAM/scripts/train_light.py
echo "== scheduler build =="
grep -n "scheduler|cosine|num_training_steps|create_optimizer" /workspace/FasterWAM/src/fasterwam/trainer.py | head -40
echo "== FW_SKIP_STATE =="
grep -rn "FW_SKIP_STATE" /workspace/FasterWAM --include=*.py | head
echo "== DONE =="