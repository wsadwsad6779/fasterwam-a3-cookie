export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/_r5_stage
rm -rf /workspace/a3_eval/replay_smoke_single /workspace/a3_eval/replay_smoke_bundle
python3 replay_forces.py --mode single --seeds 7001 7002 --out /workspace/a3_eval/replay_smoke_single
echo '--- bundle ---'
python3 replay_forces.py --mode bundle --seeds 7001 --max-steps 2000 --out /workspace/a3_eval/replay_smoke_bundle
