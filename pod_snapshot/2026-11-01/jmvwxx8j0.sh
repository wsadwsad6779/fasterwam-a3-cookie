cd /workspace/_r5_stage
python3 - <<EOF
ls = [len(x) for x in open("build_corpus_manifest.py", "rb").read().splitlines()]
print(len(ls), sum(ls))
print(",".join(str(x) for x in ls))
EOF