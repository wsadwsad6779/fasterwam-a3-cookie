cd /workspace
grep -rn 'FASTERWAM_A3_CKPT' --include=*.py . 2>/dev/null | grep -v ipynb | head -8
echo RUNMATRIX_CKPT
grep -nE 'ckpt|adapter|merged' a3_eval/run_matrix.py 2>/dev/null | head -12
find . -maxdepth 3 -name run_matrix.py