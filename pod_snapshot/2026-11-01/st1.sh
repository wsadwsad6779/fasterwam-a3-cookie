cd /workspace
echo "== r5 dir"
ls -1 a3_train_runs/a3_lora_r5 2>/dev/null
echo "== light ckpt"
ls -1 a3_train_runs/a3_lora_r5/checkpoints/light 2>/dev/null
echo "== trainer_state"
python3 -c "import json;d=json.load(open('/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/trainer_state.json'));print({k:d[k] for k in list(d)[:12]})" 2>&1 | tail -3
echo "== log tail"
for f in a3_train_runs/a3_lora_r5/*.log a3_train_runs/a3_lora_r5/*.txt a3_train_runs/a3_lora_r5/log/*; do [ -f "$f" ] && echo "--$f--" && tail -c 600 "$f"; done
echo "== procs"
ps -eo pid,etime,cmd | grep -E "train_light_r5|watch|r5metrics" | grep -v grep
echo "== eval dirs"
ls -1 a3_eval | tail -25
echo "== df"
df -h /workspace | tail -1