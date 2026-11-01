cd /workspace/_r5_stage
rm -f acc.b64
rm -rf __pycache__/replay_correct.cpython-*.pyc
sha256sum replay_correct.py
wc -lc replay_correct.py
python3 -m py_compile replay_correct.py && echo COMPILE_OK
grep -nE "failure_reason|max_pad_force|self.finished|self.failed|def reset|def act|variant_index|target_grip_force|_select_batch" replay_forces.py | head -30
echo __END__