cd /workspace/_r5_stage
python3 - <<EOF
ls = open("build_corpus_manifest.py", encoding="utf-8").read().splitlines()
for i in (59, 60, 61):
    print(i + 1, len(ls[i]), ascii(ls[i]))
EOF