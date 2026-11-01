#!/usr/bin/env python3
"""Prove the pod code snapshot survived tar -> gzip -> browser download -> extract.

Every git_only row in upload_batches.json is re-hashed from the extracted tree
and compared with the sha256 recorded while the file was still only on the pod.
"""
import hashlib
import json
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(BASE, "_r6_stage", "out", "_gitx")
d = json.load(open(os.path.join(BASE, "upload_batches.json"), encoding="utf-8"))
rows = [r for r in d["rows"] if r["action"] == "git_only"]

ok = miss = bad = 0
extra = []
seen = set()
for r in rows:
    rel = r["pod_path"][len("/workspace/"):]
    p = os.path.join(ROOT, rel.replace("/", os.sep))
    seen.add(os.path.normcase(p))
    if not os.path.exists(p):
        miss += 1
        print("MISSING", rel)
        continue
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() == r["sha256"]:
        ok += 1
    else:
        bad += 1
        print("SHA_MISMATCH", rel, h.hexdigest()[:16], "!=" , r["sha256"][:16])

for dirpath, _, files in os.walk(ROOT):
    for fn in files:
        p = os.path.normcase(os.path.join(dirpath, fn))
        if p not in seen:
            extra.append(p)

total = sum(r["bytes"] for r in rows)
print("rows=%d ok=%d missing=%d sha_bad=%d bytes=%d" % (len(rows), ok, miss, bad, total))
print("extracted_files=%d unexpected=%d" % (ok + bad + miss, len(extra)))
json.dump({"rows": len(rows), "ok": ok, "missing": miss, "sha_bad": bad, "bytes": total,
           "tar_gz_on_pod": "git_snapshot_20261101.tar.gz",
           "tar_gz_sha256": "53fc19b33396ee96e3448e556beffd2a4fa799131d5404c7b213aafb80cb8b4c"},
          open(os.path.join(BASE, "_r6_stage", "out", "git_snapshot_verify.json"), "w"), indent=1)
sys.exit(0 if (ok == len(rows) and not miss and not bad and not extra) else 1)
