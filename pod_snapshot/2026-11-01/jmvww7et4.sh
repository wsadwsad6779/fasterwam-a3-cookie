cd /workspace/_r5_stage
sha256sum replay_wrongclosure.py
wc -c replay_wrongclosure.py
python3 -m py_compile replay_wrongclosure.py && echo COMPILE_OK
grep -n "GRIPPER_ACT_DIM\|RELEASE_ACT_DIM6\|OBSERVE_FRAMES =" replay_wrongclosure.py