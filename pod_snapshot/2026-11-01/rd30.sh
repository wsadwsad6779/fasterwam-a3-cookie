cd /workspace
pgrep -af replay_forces.py | head -5
pgrep -af train_light_r5.py | head -3
pgrep -af r5_watch.sh | head -3
tail -n 3 /workspace/a3_train_runs/a3_lora_r5/train.log
ls -1 /workspace/a3_eval | grep replay
for d in replay_single_gate10 replay_bundle_dist replay_single_10clamp; do echo "== $d"; ls -1 /workspace/a3_eval/$d 2>/dev/null | head -20; done
df -BG /workspace | tail -1
