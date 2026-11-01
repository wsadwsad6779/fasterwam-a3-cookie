cd /workspace/_r5_stage
date
python3 replay_correct.py --seeds 7401 7402 --variants 2 3 --noise 0.0003 --retries 1 --max-steps 1600 --out /workspace/a3_eval/demos_correct_smoke 2>&1 | tail -c 4000
date
echo __END__