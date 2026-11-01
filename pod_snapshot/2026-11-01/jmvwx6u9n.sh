cd /workspace/_r5_stage
grep -n anomaly_kind replay_correct.py
echo == KINDS
for d in /workspace/a3_eval/demos_correct_c*; do echo -- $d; grep -o empty_grasp $d/summary.json | wc -l; grep -o stuck_no_force $d/summary.json | wc -l; grep -o excessive_pad_force $d/summary.json | wc -l; done