#!/bin/bash
grep --version | head -1
echo "== record region =="
awk "NR>=230 && NR<=270 {print NR\": "$0}" " /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== resolve_control head =="
awk "NR>=85 && NR<=115 {print NR\": "$0}" " /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== docstring =="
awk "NR>=1 && NR<=35 {print NR\": "$0}" " /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== state setup =="
awk "NR>=145 && NR<=172 {print NR\": "$0}" " /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py
echo "== DONE =="