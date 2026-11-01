set -u
source /workspace/activate_fasterwam_a3.sh
python3 - <<PY
p="/workspace/_r4_stage/mech_eval.py"
t=open(p).read().replace("    rigid = np.zeros((T, N))","    rigid = np.zeros((T, N), dtype=bool)")
open(p,"w").write(t)
print("patched", t.count("dtype=bool"))
PY
python3 -u /workspace/_r4_stage/mech_eval.py 2>&1 | tail -140