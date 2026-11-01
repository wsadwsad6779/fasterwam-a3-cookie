import json, os, collections
R="/workspace/a3_data/a3-front-close-left-100"
for f in ["collection_summary.json"]:
    print("=== ", f); print(open(os.path.join(R,f)).read()[:3000])
for f in ["a3_episode_metadata.jsonl","attempts.jsonl"]:
    p=os.path.join(R,f); ls=[json.loads(x) for x in open(p) if x.strip()]
    print("===",f,"n",len(ls))
    print("keys", sorted(ls[0].keys()))
    print("sample", json.dumps(ls[0])[:600])
    # aggregate booleans/success-like keys
    for k in ls[0]:
        vals=[r.get(k) for r in ls]
        if all(isinstance(v,(bool,int,float,str)) or v is None for v in vals):
            c=collections.Counter(map(str,vals))
            if len(c)<=6: print("  ",k,dict(c))
# meta info
for f in ["meta/info.json","README.md"]:
    p=os.path.join(R,f)
    if os.path.exists(p): print("===",f); print(open(p).read()[:1500])

