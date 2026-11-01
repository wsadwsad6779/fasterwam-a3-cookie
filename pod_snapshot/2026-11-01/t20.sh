sed -n '362,395p' /workspace/a3_dual_arm_sim/src/a3_dual_arm_sim/sim/env.py
grep -rnE 'proprio_output_dim|proprio' /workspace/FasterWAM/configs/data/a3_2cam.yaml | head -12
echo ==PILOT==
du -sh /workspace/a3_eval/pilot_single_lowforce_clean /workspace/a3_eval/pilot_single_lowforce_noisy
tail -c 300 /workspace/pilotA.out
echo DONE