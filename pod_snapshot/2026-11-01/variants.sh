export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
while pgrep -f replay_forces.py > /dev/null; do sleep 10; done
pgrep -af demo30b
sleep 3
python3 /workspace/_r5_stage/patch_variants.py
sha256sum /workspace/_r5_stage/replay_forces.py
python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise 0.0003 --seeds 7041 --out /workspace/a3_eval/demos_postpatch_smoke
