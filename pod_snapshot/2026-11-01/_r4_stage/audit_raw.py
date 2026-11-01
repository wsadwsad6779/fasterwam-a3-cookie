import json, glob, os
D = "/workspace/data/a3_front_close_left_100"
print("== meta dir ==")
for p in sorted(glob.glob(D + "/meta/*"))[:20]:
    print(os.path.basename(p), os.path.getsize(p))
ip = D + "/meta/info.json"
if os.path.exists(ip):
    info = json.load(open(ip))
    print("info keys:", list(info.keys()))
    print("episodes", info.get("total_episodes"), "frames", info.get("total_frames"), "fps", info.get("fps"), "ver", info.get("codebase_version"))
    feats = info.get("features", {})
    print("== features ==")
    for k, v in feats.items():
        dt = v.get("dtype")
        sh = v.get("shape")
        nm = v.get("names") or []
        print(k.ljust(34), str(dt).ljust(10), sh, "names=" + str(len(nm)))
    needles = ("force", "wrench", "torque", "sensor", "contact", "timestamp", "time")
    fk = [k for k in feats if any(s in k.lower() for s in needles)]
    print("force/time-ish:", fk)
else:
    print("no meta/info.json")
tp = D + "/meta/tasks.jsonl"
print("== tasks ==")
print(open(tp).read()[:300] if os.path.exists(tp) else "none")
df = sorted(glob.glob(D + "/data/**/*.parquet", recursive=True)) or sorted(glob.glob(D + "/data/*/*"))
print("== data files ==", len(df))
print(df[:3])
if df:
    import pyarrow.parquet as pq
    t = pq.read_table(df[0])
    print("columns:", t.column_names)
    print("rows", t.num_rows)
    for f in t.schema:
        print("  ", f.name, f.type)
