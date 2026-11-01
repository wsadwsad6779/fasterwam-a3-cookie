import json, sys, pathlib
rows = []
for d in sys.argv[1:]:
    p = pathlib.Path(d) / "adjudication_c3.json"
    if not p.exists():
        print(d, "NOFILE"); continue
    j = json.loads(p.read_text())
    ent = j["episodes"] if isinstance(j, dict) else j
    acc = [e["seed"] for e in ent if e.get("strict_accept")]
    rej = [e["seed"] for e in ent if not e.get("strict_accept")]
    print(pathlib.Path(d).name, "n=" + str(len(ent)), "accept=" + str(len(acc)), acc, "reject=" + str(rej))
    rows += acc
print("TOTAL", len(rows), len(set(rows)))
print("demoted_in_c3", [s for s in (7243,7250,7260) if s in rows])