#!/bin/bash
{ ps -eo pid,etime,pcpu,rss,cmd | grep -v grep | grep -E "python3|run_matrix"; echo ---; rocm-smi --showuse 2>/dev/null | head -20; } > /workspace/_r4_stage/ps.txt 2>&1
echo __PS__ >> /workspace/_r4_stage/ps.txt