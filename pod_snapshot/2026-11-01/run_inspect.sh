#!/usr/bin/env bash
set -x
source /workspace/activate_fasterwam_a3.sh >/dev/null 2>&1
cd /workspace/fasterwam_a3_bridge/bridge
python3 /workspace/inspect_held.py
