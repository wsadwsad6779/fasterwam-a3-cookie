find /workspace -maxdepth 5 -name prepare_a3_arrays.py 2>/dev/null | head
echo ===
grep -rn --include=*.py -l observation.state /workspace/FasterWAM/scripts /workspace/FasterWAM/src/fasterwam 2>/dev/null | head -10
echo ===
ls /workspace/FasterWAM/scripts/ds_configs | head -30