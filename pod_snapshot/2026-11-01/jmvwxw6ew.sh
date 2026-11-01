cd /workspace/_r5_stage
wc -c build_corpus_manifest.py
sha256sum build_corpus_manifest.py
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -m py_compile build_corpus_manifest.py && echo COMPILE_OK
head -3 /workspace/fin2.out 2>/dev/null
grep -c ^{ /workspace/c6a.out /workspace/c7a.out