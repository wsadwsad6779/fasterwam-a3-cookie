tail -6 /workspace/a3_train_runs/a3_lora_r5/train.log
echo CKPT
ls -1t /workspace/a3_train_runs/a3_lora_r5/checkpoints | head -6
echo PS
ps -eo pid,etime,cmd | grep train_a3 | grep -v grep