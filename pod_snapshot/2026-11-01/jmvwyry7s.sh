tail -12 /workspace/v1_live.out
echo GREP2
grep -o 'force[a-z_0-9]*' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/env.py | sort -u
echo MIRROR
md5sum /workspace/fasterwam-a3-cookie/src/a3_dual_arm_sim/env.py 2>/dev/null || echo NO_MIRROR_COPY