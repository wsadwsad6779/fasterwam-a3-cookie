#!/bin/bash
set -u
echo "== sha =="
sha256sum /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== argparse =="
grep -n "add_argument" /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py | head -40
echo "== controls block =="
awk "/--controls/,/--seeds/" /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== else branch =="
grep -n "post_w = pre.copy" -A4 -B4 /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== step call =="
grep -n "env.step(post)" -A3 /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== record =="
grep -n "safety_layer|estop_steps|binfloor_steps|duration_s" /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== disk =="
df -h /workspace | tail -1
echo "== terminals =="
ls /workspace/_r4_stage | head -30
echo "== DONE =="