cd /workspace/_r5_stage
wc -c build_corpus_manifest.py
sha256sum build_corpus_manifest.py
python3 -m py_compile build_corpus_manifest.py && echo COMPILE_OK
tail -3 /workspace/fin2.out 2>/dev/null
ps -eo pid,cmd | grep -E replay_correct | grep -v grep