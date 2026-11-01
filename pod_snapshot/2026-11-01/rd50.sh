cd /workspace
tail -n 1 /workspace/a3_train_runs/a3_lora_r5/train.log
python3 - <<PY
import json
s=json.load(open("/workspace/a3_eval/replay_bundle_dist/summary.json"))
print("bundle episodes",len(s))
for r in s:
  cl=r["clamps"]
  print(r["seed"],r["steps"],r["stop"][:20],"inb",r["in_target_final"],"up",r["moved_into_bin_upright"],"tilt",r["moved_into_bin_tilted"],"o2",r["frames_over_2n"],"pmax",round(r["pad_max_n"],3),"clamps",len(cl),"hold",round(sum(c["hold_mean_pad_n"] for c in cl)/len(cl),3),"carrymax",round(max(c["carry_peak_pad_n"] or 0 for c in cl),3))
PY
ls -la /workspace/a3_eval | tail -12
