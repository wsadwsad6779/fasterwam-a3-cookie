export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
df -B1 /workspace | tail -1
python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise 0.0003 --seeds 7201 7202 7203 7204 7205 7206 7207 7208 7209 7210 7211 7212 7213 7214 7215 7216 7217 7218 7219 7220 7221 7222 7223 7224 7225 7226 7227 7228 7229 7230 --out /workspace/a3_eval/demos_single_v30
df -B1 /workspace | tail -1
echo DEMOS30V_COMPLETE
