tail -c 400 /workspace/a3_train_runs/a3_lora_r5/train.log 2>/dev/null | tail -3
ls -1 /workspace/a3_train_runs/a3_lora_r5/checkpoints/light 2>/dev/null | tail -4
date
grep -c . /workspace/cc1.out
tail -c 900 /workspace/cc1.out
echo __END__
