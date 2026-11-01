set -u
source /workspace/activate_fasterwam_a3.sh
sha256sum /workspace/_r4_stage/mech_eval.py
wc -c /workspace/_r4_stage/mech_eval.py
python3 -u /workspace/_r4_stage/mech_eval.py 2>&1 | tail -120