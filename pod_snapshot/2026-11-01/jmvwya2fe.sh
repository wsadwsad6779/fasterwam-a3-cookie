cd /workspace/a3_eval
sha256sum corpus_manifest_v1.json
python3 -c "import json,os,collections;d=json.load(open(0));e=d["episodes"] if "episodes" in d else d["entries"] if "entries" in d else None;print(list(d.keys()));print(len(e))" < corpus_manifest_v1.json