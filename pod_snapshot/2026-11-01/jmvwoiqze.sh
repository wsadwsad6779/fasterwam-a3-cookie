set -u
head -48 /workspace/_r4_stage/mech_eval.out
echo ====TRAINLIGHT
cat -v /workspace/FasterWAM/scripts/train_light.py | head -80
echo ====R4DIR
ls -la /workspace/a3_train_runs/a3_lora_r4/