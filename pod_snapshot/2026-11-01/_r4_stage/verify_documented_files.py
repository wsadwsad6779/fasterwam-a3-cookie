"""Verify the sizes/hashes transcribed into PRE_R4_CHECK.md against the cloud.

Compares check.tsv against the pre-computed /workspace/a3_eval/disk_hashes.txt
so that no large file has to be re-read. Anything missing from that inventory
and under 300 MB is hashed live; larger ones are reported, not hashed.

Run on the A3 pod:
    /opt/venv/bin/python3 verify_documented_files.py /workspace/_r4_stage/check.tsv
"""
import hashlib
import os
import sys

INVENTORY = "/workspace/a3_eval/disk_hashes.txt"
LIVE_HASH_LIMIT = 300 * 1024 * 1024


def load_inventory():
    out = {}
    with open(INVENTORY) as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                out[parts[1].strip()] = parts[0].lower()
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(list_file):
    inv = load_inventory()
    failures = []
    for line in open(list_file):
        line = line.rstrip("\n")
        if not line:
            continue
        mode, path, size, want = (line.split("\t") + ["", "", "", ""])[:4]
        size = int(size)
        if not os.path.exists(path):
            print("MISSING %s" % path)
            failures.append(path)
            continue
        real = os.path.getsize(path)
        if real != size:
            print("SIZEFAIL %s documented=%d actual=%d" % (path, size, real))
            failures.append(path)
            continue
        if mode == "SIZEONLY":
            print("SIZEOK %s" % path)
            continue
        have = inv.get(path)
        if have is None:
            if size > LIVE_HASH_LIMIT:
                print("NOTINVENTORYED %s size=%d" % (path, size))
                failures.append(path)
                continue
            have = sha256(path)
        if have == want:
            print("HASHOK %s" % path)
        else:
            print("HASHFAIL %s documented=%s actual=%s" % (path, want, have))
            failures.append(path)
    print("RESULT %d checked, %d failed" %
          (sum(1 for l in open(list_file) if l.strip()), len(failures)))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1])
