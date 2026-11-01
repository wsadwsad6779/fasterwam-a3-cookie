cd /workspace/_r5_stage
python3 -m py_compile probe_release.py && echo COMPILE_OK
PYTHONPATH=/workspace/a3_dual_arm_sim/src python3 probe_release.py 2>&1 | sed -n "1,30p;82,112p"