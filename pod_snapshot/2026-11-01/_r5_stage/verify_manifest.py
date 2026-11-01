import json, sys, pathlib, collections
d = json.loads(pathlib.Path(sys.argv[1]).read_text())
print("keys", sorted(d.keys()))
ent = d["entries"]
print("n", len(ent))
print("family", dict(collections.Counter(e["family"] for e in ent)))
seeds = [(e["dir"], e["npz"]) for e in ent]
dup = [k for k, v in collections.Counter(seeds).items() if v > 1]
print("dupes", dup)
missing = [s for s in seeds if not (pathlib.Path(s[0]) / s[1]).exists()]
print("missing_npz", len(missing), missing[:5])
print("total_bytes", sum(e["npz_bytes"] for e in ent))
print("sha_ok", sum(1 for e in ent if e["npz_sha256"]), "of", len(ent))
print("tally", d["tally"])
print('--- volume ---')
KB = 281 * 1024
by = collections.defaultdict(lambda: [0, 0])
for e in ent:
    st = int(e["steps"] or 0)
    by[e["family"]][0] += 1
    by[e["family"]][1] += st
tot = 0
for f in sorted(by):
    n, fr = by[f]
    tot += fr
    print(f, "n=" + str(n), "frames=" + str(fr), "avg=" + str(round(fr / max(n, 1), 1)), "GB=" + str(round(fr * KB / 1024 ** 3, 2)))
print("ALL frames=" + str(tot), "GB=" + str(round(tot * KB / 1024 ** 3, 2)))
print("minmax_steps", min(int(e["steps"] or 0) for e in ent), max(int(e["steps"] or 0) for e in ent))
