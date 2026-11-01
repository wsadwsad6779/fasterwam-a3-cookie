cd /workspace/_r5_stage
python3 - <<EOF
d = open("build_corpus_manifest.py", "rb").read()
ls = [len(x) for x in d.split(b"
")]
print(len(ls), sum(ls))
print(",".join(str(x) for x in ls))
EOF