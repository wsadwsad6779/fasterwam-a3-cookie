grep -o empty_grasp /workspace/c6a.out | wc -l
grep -o stuck_no_force /workspace/c6a.out | wc -l
grep -o excessive_pad_force /workspace/c6a.out | wc -l
grep -oE "seed.: 72.., .variant.: [0-9]+" /workspace/c6a.out | head -8
grep -oE ".anomaly_observed.: (true|false)" /workspace/c6a.out | head -8