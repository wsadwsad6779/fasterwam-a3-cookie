cd /workspace
python3 - <<PY
import json
s=json.load(open("/workspace/a3_eval/replay_single_gate/summary.json"))
print("episodes",len(s))
for r in s:
  print(r["seed"],r["steps"],r["stop"],"inb",r["in_target_final"],"up",r["moved_into_bin_upright"],"tilt",r["moved_into_bin_tilted"],"p50",r["pad_p50_n"],"p95",r["pad_p95_n"],"mx",r["pad_max_n"],"o2",r["frames_over_2n"])
  for c in r["clamps"]:
    print("   cl",round(c["closure_peak_pad_n"],3),"hold",round(c["hold_mean_pad_n"],3),"carry",round(c["carry_peak_pad_n"] or 0,3),"post",round(c["post_release_pad_n"],3),"n",c["max_cookies_at_grip"],"raised",c["cookies_raised_20mm"],"lift",c["carried_lift_mm"],"xy",int(c["placed_in_bin_xy"]),"up",int(c["placed_upright"]),"acc",int(c["accept"]))
print("settings0",json.dumps(s[0]["settings"]))
PY
grep -n clamps /workspace/gate.out | tail -5
head -c 600 /workspace/gate.out
