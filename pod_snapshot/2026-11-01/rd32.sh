cd /workspace
python3 - <<PY
import json,statistics as st
s=json.load(open("/workspace/a3_eval/replay_single_gate10/summary.json"))
ak=[r for r in s if r["clamps"] and all(c["accept"] for c in r["clamps"])]
print("episodes",len(s),"accept_all",len(ak))
for r in s:
  for c in r["clamps"]:
    print(r["seed"], r["steps"], r["stop"], "acc",int(c["accept"]), "grip",c["closure_peak_pad_n"], "hold",c["hold_mean_pad_n"], "carry",c["carry_peak_pad_n"], "post",c["post_release_pad_n"], "n",c["max_cookies_at_grip"], "lift",c["carried_lift_mm"], "dist",c["carried_distance_mm"], "xy",int(c["placed_in_bin_xy"]), "up",int(c["placed_upright"]), "over2",r["frames_over_2n"], "pmax",r["pad_max_n"])
vals=lambda k:[c[k] for r in s for c in r["clamps"] if c[k] is not None]
for k in ["first_contact_pad_n","closure_peak_pad_n","hold_mean_pad_n","carry_peak_pad_n","pre_release_pad_n","post_release_pad_n"]:
  v=vals(k); print(k,"n",len(v),"min",round(min(v),3),"med",round(st.median(v),3),"max",round(max(v),3))
print("wrist p50 med",round(st.median([r["wrist_p50_n"] for r in s]),3))
print("lift med",round(st.median([c["carried_lift_mm"] for r in s for c in r["clamps"] if c["carried_lift_mm"] is not None]),1))
print("dist med",round(st.median([c["carried_distance_mm"] for r in s for c in r["clamps"] if c["carried_distance_mm"] is not None]),1))
PY
