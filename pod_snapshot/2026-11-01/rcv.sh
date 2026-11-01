set -x
cd /workspace/_r5_stage
sha256sum replay_correct.py
wc -c replay_correct.py
python3 -m py_compile replay_correct.py && echo COMPILE_OK
grep -nE "^(FORCE_KEY|L_INNER|L_OUTER|IDLE_PHASES|PHASE_NAMES|BIN_XY|BIN_INNER_HALF|BIN_HALF|def make_env|def clamp_stats|def segments|class SingleCookieLowForce)" replay_forces.py
grep -c applied_action replay_forces.py
sha256sum replay_forces.py
df -h /workspace | tail -1
ls -1 /workspace/a3_eval | tail -20