#!/bin/bash
# Sequential transport matrix. One process per case on purpose (see probe4 docstring).
set -u
cd /workspace/_v1_stage
echo "### hosts"; python3 hf_write_probe4.py --hosts
for c in xet_8m xet_70m regular_8m regular_70m; do
  echo "### case $c"
  timeout 1200 python3 hf_write_probe4.py --case "$c" || echo "CASE_FAILED $c rc=$?"
done
echo "### case regular_200m"
timeout 1800 python3 hf_write_probe4.py --case regular_200m || echo "CASE_FAILED regular_200m rc=$?"
echo "### collect"
python3 hf_write_probe4.py --collect
echo DONE_PROBE4
