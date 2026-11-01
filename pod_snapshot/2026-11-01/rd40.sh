cd /workspace/a3_train_runs/a3_lora_r5
ls -la
ls -la checkpoints/light
echo ====PREFLIGHT
cat preflight.txt
echo ====LRRESTART
cat lr_restart.json
echo ====RESUME_LINES
grep -n -e resumed -e sampler -e lr_restored -e restart -e trainable -e base train.log | head -30
echo ====TAIL
tail -n 4 train.log
