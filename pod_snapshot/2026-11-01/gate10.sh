export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/_r5_stage
python3 replay_forces.py --mode single --clamps 1 --noise 0.0003 --max-steps 900 --seeds 7041 7042 7043 7044 7045 7046 7047 7048 7049 7050 --out /workspace/a3_eval/replay_single_gate10
ls -la /workspace/a3_eval/replay_single_gate10 | head -5
tail -c 400 /workspace/gate.out
