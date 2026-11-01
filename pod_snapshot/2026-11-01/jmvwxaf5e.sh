head -4 /workspace/fin1.out
grep -n V9ADJ /workspace/fin1.out
awk "/V9ADJ/,/C4C5ADJ/" /workspace/fin1.out