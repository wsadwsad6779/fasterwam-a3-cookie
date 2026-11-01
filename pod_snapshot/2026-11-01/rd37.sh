cd /workspace
python3 - <<PY
import json
s=json.load(open("/workspace/a3_eval/replay_single_gate/summary.json"))
g=lambda d,k,dft=None: d.get(k,dft)
for r in s:
  print(r["seed"],r["steps"],r["stop"][:52],"inb",r["in_target_final"],"up",r["moved_into_bin_upright"],"mx",round(r["pad_max_n"],3),"o2",r["frames_over_2n"])
  for c in r["clamps"]:
    print("   cl",round(g(c,"closure_peak_pad_n",0),3),"carry",round(g(c,"carry_peak_pad_n") or 0,3),"post",round(g(c,"post_release_pad_n",0),3),"n",g(c,"max_cookies_at_grip"),"raised",g(c,"cookies_raised_20mm","NA"),"lift",g(c,"carried_lift_mm"),"xy",int(bool(g(c,"placed_in_bin_xy"))),"up",int(bool(g(c,"placed_upright"))),"acc",int(bool(g(c,"accept"))))
print("S",json.dumps(s[0]["settings"]))
PY
grep -n "###" /workspace/gate.out
ls -la /workspace/a3_eval/replay_bundle_dist/
du -sh /workspace/a3_eval/r4_b0_c0_paired /workspace/a3_eval/r4_c0_frozen 2>/dev/null
