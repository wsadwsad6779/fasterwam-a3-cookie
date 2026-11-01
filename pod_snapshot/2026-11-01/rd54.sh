grep -c accept /workspace/demos30v.out 2>/dev/null
tail -c 300 /workspace/demos30v.out 2>/dev/null
tail -n 1 /workspace/a3_train_runs/a3_lora_r5/train.log
python3 - <<PY
import json,os
p="/workspace/a3_eval/demos_single_v30/summary.json"
if os.path.exists(p):
  s=json.load(open(p))
  print("eps",len(s))
  for r in s:
    c=r["clamps"][0] if r["clamps"] else {}
    print(r["seed"],"acc",int(bool(c.get("accept"))),"idx",c.get("carried_index"),"closure",round(c.get("closure_peak_pad_n") or 0,3),"lift",c.get("carried_lift_mm"),"dist",round(c.get("carried_distance_mm") or 0,1),"o2",r["frames_over_2n"],"inb",r["in_target_final"],"stop",str(r["stop"])[:18])
PY
