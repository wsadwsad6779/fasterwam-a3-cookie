set -u
source /workspace/activate_fasterwam_a3.sh
sha256sum /workspace/_r4_stage/mech_eval.py
python3 -u /workspace/_r4_stage/mech_eval.py > /workspace/_r4_stage/mech_eval.out 2>&1; echo rc=$?
tail -70 /workspace/_r4_stage/mech_eval.out