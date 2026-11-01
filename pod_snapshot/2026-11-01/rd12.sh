cd /workspace
python3 - <<'PYEOF'
import json
m = json.load(open('/workspace/a3_eval/r4_b0_c0_paired/manifest_lora.json'))
print(json.dumps(m, indent=1)[:2600])
PYEOF
ls /workspace/_r5_stage/
ls /workspace/a3_eval/r4_b0_c0_paired/ | head
ls /workspace/a3_eval/r4_c0_frozen/ | head
tail -2 /workspace/a3_train_runs/a3_lora_r5/train.log
