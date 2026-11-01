#!/bin/bash
echo "== frozen dir =="
ls -la /workspace/a3_eval/r4_c0_frozen 2>&1 | head -40
echo "== manifest =="
cat /workspace/a3_eval/r4_c0_frozen/manifest_lora.json 2>&1
echo "== summary =="
cat /workspace/a3_eval/r4_c0_frozen/summary_lora.json 2>&1
echo "== drivers =="
ls /workspace/_r4_stage/*.sh 2>&1
echo "== grep run_matrix =="
grep -l run_matrix /workspace/_r4_stage/*.sh 2>&1
echo "== gpu =="
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv 2>&1 | tail -3
ps -eo pid,etime,pcpu,args | grep -i run_matrix | grep -v grep | head
echo "== DONE =="