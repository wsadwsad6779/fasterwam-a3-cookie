tail -c 300 /workspace/demo30b.out
ls /workspace/a3_eval | grep demos
python3 - <<PY
import json
for d in ["demos_single_n0","demos_single_n3","demos_single_n6"]:
  try: s=json.load(open("/workspace/a3_eval/"+d+"/summary.json"))
  except Exception as e: print(d,"pending",e); continue
  ak=sum(1 for r in s if r["clamps"] and all(c.get("accept") for c in r["clamps"]))
  idx=sorted(set(c.get("carried_index") for r in s for c in r["clamps"]))
  print(d,"eps",len(s),"accept",ak,"carried_idx",idx,"noise",s[0]["noise_m"],"padmax",[round(r["pad_max_n"],3) for r in s])
PY
