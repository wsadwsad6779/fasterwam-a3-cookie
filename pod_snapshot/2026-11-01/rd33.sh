cd /workspace
tail -c 1200 /workspace/gate.out
echo ====BUNDLE7031
python3 - <<PY
import json
s=json.load(open("/workspace/a3_eval/replay_bundle_dist/summary.json"))
print(len(s))
for r in s:
  print(r["seed"],r["steps"],r["stop"],"inb",r["in_target_final"],"up",r["moved_into_bin_upright"],"tilt",r["moved_into_bin_tilted"],"p50",r["pad_p50_n"],"p95",r["pad_p95_n"],"mx",r["pad_max_n"],"o2",r["frames_over_2n"],"o25",r["frames_over_25n"],"wr",r["wrist_p50_n"])
  for c in r["clamps"]:
    print("   ","fc",c["first_contact_pad_n"],"cl",c["closure_peak_pad_n"],"hold",c["hold_mean_pad_n"],"carry",c["carry_peak_pad_n"],"pre",c["pre_release_pad_n"],"post",c["post_release_pad_n"],"n",c["max_cookies_at_grip"],"raised",c["cookies_raised_20mm"],"lift",c["carried_lift_mm"],"dist",c["carried_distance_mm"],"xy",int(c["placed_in_bin_xy"]),"up",int(c["placed_upright"]),"acc",int(c["accept"]))
PY
