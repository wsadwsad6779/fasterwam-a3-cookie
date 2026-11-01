cd /workspace/_r5_stage
cat > r5_eval.sh <<'SEOF'
#!/usr/bin/env bash
# r5-vs-r4 paired B0/C0 evaluation; same protocol as r4_paired.sh, new output dir.
set -u
source /workspace/activate_fasterwam_a3.sh
export FASTERWAM_A3_INSTRUCTION='transfer 10 cookies into target box'
cd /workspace/fasterwam_a3_bridge/bridge/_next || exit 1
A=/workspace/a3_train_runs/a3_lora_r5/checkpoints/light/step_008000/adapter.pt
O=/workspace/a3_eval/r5_b0_c0_paired
L=/workspace/a3_eval/r5_b0_c0_paired.log
if [ ! -f "$A" ]; then echo "missing $A" > "$L"; exit 1; fi
if [ -e "$O" ] && [ -n "$(ls -A "$O" 2>/dev/null)" ]; then echo "$O is not empty, refusing" > "$L"; exit 1; fi
FREE=$(df -B1 --output=avail /workspace | tail -1)
if [ "$FREE" -lt 2000000000 ]; then echo "only $FREE bytes free, refusing" > "$L"; exit 1; fi
mkdir -p "$O"
{
  date -u +%FT%TZ
  echo '== paired B0/C0, r5 step_008000 adapter on r3 merged base, seeds 11 21 22, replan 10, 600 steps, video on'
  sha256sum run_matrix.py "$A"
  echo free_bytes=$FREE
  python3 run_matrix.py --model lora --ckpt /workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt --adapter "$A" --stats /workspace/a3_data/a3_fw/dataset_stats.json --gripper-mode pm1 --controls B0,C0 --seeds 11 21 22 --replan 10 --max-steps 600 --position-noise 0 --yaw-noise 0 --batch-offset 0 --video --out "$O"
  echo eval_exit=$?
  ls -la "$O"
  df -h /workspace | tail -1
  date -u +%FT%TZ
} > "$L" 2>&1
echo __R5_EVAL_DONE__ >> "$L"
SEOF
cat > r5_watch.sh <<'WEOF'
#!/usr/bin/env bash
# Hold until the r5 training process exits, then run the paired evaluation once.
W=/workspace/_r5_stage/r5_watch.log
echo "watch start $(date -u +%FT%TZ)" > "$W"
while pgrep -f train_light_r5.py > /dev/null; do sleep 60; done
echo "training exited $(date -u +%FT%TZ)" >> "$W"
tail -4 /workspace/a3_train_runs/a3_lora_r5/train.log >> "$W"
bash /workspace/_r5_stage/r5_eval.sh
echo "r5_eval.sh exit=$? $(date -u +%FT%TZ)" >> "$W"
WEOF
bash -n r5_eval.sh && bash -n r5_watch.sh && echo SYNTAX_OK
sha256sum r5_eval.sh r5_watch.sh
setsid nohup bash /workspace/_r5_stage/r5_watch.sh >/dev/null 2>&1 < /dev/null &
sleep 2
pgrep -af r5_watch.sh
