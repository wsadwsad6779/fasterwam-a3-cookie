set -e
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
df -B1 /workspace | tail -1
for cfg in "0:no 7101 7102 7103 7104 7105 7106 7107 7108 7109 7110" "3:0003 7111 7112 7113 7114 7115 7116 7117 7118 7119 7120" "6:0006 7121 7122 7123 7124 7125 7126 7127 7128 7129 7130"; do
  nm=${cfg%%:*}; nz=${cfg#*:}
  if [ "$nz" = "no" ]; then NOISE=0; else NOISE=0.00$nz; fi
  O=/workspace/a3_eval/demos_single_$nm
  echo "=== batch $nm noise $NOISE out $O"
  df -B1 /workspace | tail -1
  python3 /workspace/_r5_stage/replay_forces.py --mode single --clamps 1 --noise $NOISE --seeds $@
  echo skip
done
