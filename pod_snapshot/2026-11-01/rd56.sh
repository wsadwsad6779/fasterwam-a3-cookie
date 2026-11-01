tail -c 200 /workspace/demos30v.out
python3 - <<PY
import json,os
p="/workspace/a3_eval/demos_single_v30/summary.json"
s=json.load(open(p))
print("eps",len(s))
ok=0
for r in s:
  c=(r["clamps"] or [{}])[0]
  a=bool(c.get("accept")); ok+=a
  print(r["seed"],"acc",int(a),"idx",c.get("carried_index"),"clos",round(c.get("closure_peak_pad_n") or 0,3),"hold",round(c.get("hold_mean_pad_n") or 0,3),"carry",round(c.get("carry_peak_pad_n") or 0,3),"lift",c.get("carried_lift_mm"),"dist",round(c.get("carried_distance_mm") or 0,1),"o2",r["frames_over_2n"],"inb",r["in_target_final"],str(r["stop"])[:26])
print("accept",ok,"/",len(s))
PY
