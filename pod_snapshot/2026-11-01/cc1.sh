echo '--- csm4 cmd ---'
grep -o 'python3 -u.*' /workspace/csm4.sh
head -c 400 /workspace/csm4.out
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
df -B1 /workspace | tail -1
python3 -u /workspace/_r5_stage/replay_correct.py --seeds 7204 7208 7228 7207 7211 7227 7215 7219 --variants 3 7 27 6 10 26 14 18 --noise 0.0003 --retries 2 --recover-variants 0 37 5 --max-steps 2400 --out /workspace/a3_eval/demos_correct_c1
date
df -B1 /workspace | tail -1
echo __END__
