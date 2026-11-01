import hashlib
import pathlib

P = pathlib.Path("/workspace/_r5_stage/replay_forces.py")
src = P.read_text(encoding="utf-8")
before = hashlib.sha256(src.encode()).hexdigest()


def sub(old, new):
    global src
    n = src.count(old)
    if n != 1:
        raise SystemExit("anchor seen %d times: %r" % (n, old[:70]))
    src = src.replace(old, new, 1)


sub(
    "        columns = sorted(set(source[:, 0].tolist()))\n"
    "        x = columns[self.variant_index % len(columns)]\n",
    "        columns = sorted(set(source[:, 0].tolist()))[:2]\n"
    "        x = columns[self.variant_index % len(columns)]\n",
)

sub(
    "        start = (self.variant_index // len(columns)) % len(order)\n",
    "        start = (self.variant_index // len(columns)) % min(10, len(order))\n",
)

P.write_text(src, encoding="utf-8")
print("sha_before", before)
print("sha_after", hashlib.sha256(src.encode()).hexdigest())
