cd /workspace/_r5_stage
sha256sum probe_release.py
python3 -m py_compile probe_release.py && echo COMPILE_OK
grep -n "if False" probe_release.py; echo GREP_DONE
python3 probe_release.py 2>&1 | head -70