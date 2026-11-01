export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/_r5_stage
echo '### 1 single x3 clamps, noisy, 10 episodes'
python3 replay_forces.py --mode single --clamps 3 --noise 0.0003 --max-steps 900 --seeds 7011 7012 7013 7014 7015 7016 7017 7018 7019 7020 --out /workspace/a3_eval/replay_single_gate
echo '### 2 bundle, noisy, 6 episodes'
python3 replay_forces.py --mode bundle --noise 0.0003 --max-steps 1400 --seeds 7031 7032 7033 7034 7035 7036 --out /workspace/a3_eval/replay_bundle_dist
echo '### 3 single x10 clamps, clean'
python3 replay_forces.py --mode single --clamps 10 --noise 0.0 --max-steps 3200 --seeds 7001 --out /workspace/a3_eval/replay_single_10clamp
df -h /workspace | tail -1
echo ALLDONE
