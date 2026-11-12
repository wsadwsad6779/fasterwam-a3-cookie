"""Reassemble a chunked filePath download and prove it matches the pod hash.

Usage: py -3.13 tko_get.py <prefix> <out> <expected_sha256>
Chunks are the JSON.stringify'd base64 slices written by evaluate_script.
"""
import base64
import hashlib
import json
import re
import sys
from pathlib import Path

prefix, out, expect = sys.argv[1], sys.argv[2], sys.argv[3]
parts = sorted(Path(".").glob(prefix + "*.txt"), key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
blob = []
for p in parts:
    t = p.read_text(encoding="utf-8")
    t = json.loads(t) if t.startswith('"') else t
    t = re.sub(r"\s+", "", t)
    if not re.fullmatch(r"[A-Za-z0-9+/=]*", t):
        raise SystemExit(f"{p}: not pure base64 ({len(t)} chars)")
    print(f"{p.name}: {len(t)} chars")
    blob.append(t)
s = "".join(blob)
data = base64.b64decode(s + "=" * (-len(s) % 4))
sha = hashlib.sha256(data).hexdigest()
print("chunks", len(parts), "b64", len(s), "bytes", len(data), "sha256", sha)
if sha != expect:
    raise SystemExit("HASH MISMATCH -> transport is not byte-exact, do not use this file")
Path(out).write_bytes(data)
print("WROTE", out)
