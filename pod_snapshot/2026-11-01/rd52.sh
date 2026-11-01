cd /workspace
python3 - <<PY
import json
s=json.load(open("/workspace/a3_eval/replay_single_10clamp/summary.json"))
for r in s:
  print("seed",r["seed"],"steps",r["steps"],"stop",str(r["stop"])[:70])
  print("inb",r["in_target_final"],"up",r["moved_into_bin_upright"],"tilt",r["moved_into_bin_tilted"],"padmax",round(r["pad_max_n"],3),"o2",r["frames_over_2n"],"wr p50",r["wrist_p50_n"])
  for i,c in enumerate(r["clamps"]):
    print(" clamp",i,"fc",round(c.get("first_contact_pad_n") or 0,3),"closure",round(c.get("closure_peak_pad_n") or 0,3),"hold",round(c.get("hold_mean_pad_n") or 0,3),"carry",round(c.get("carry_peak_pad_n") or 0,3),"post",round(c.get("post_release_pad_n") or 0,3),"n",c.get("max_cookies_at_grip"),"lift",c.get("carried_lift_mm"),"dist",c.get("carried_distance_mm"),"xy",int(bool(c.get("placed_in_bin_xy"))),"up",int(bool(c.get("placed_upright"))),"acc",int(bool(c.get("accept"))))
PY
