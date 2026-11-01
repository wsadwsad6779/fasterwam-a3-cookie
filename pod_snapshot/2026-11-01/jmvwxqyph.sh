cd /workspace/_r5_stage
python3 - <<EOF
import json
st = json.load(open("/workspace/a3_data/a3_fw_force/force_stats.json"))
print("names", st["names"])
for k in ("residual_median", "residual_mean", "residual_std", "residual_p95", "residual_absmax", "raw_p95", "raw_max"):
    v = st.get(k)
    print("%-16s" % k, [round(float(x), 4) for x in v] if isinstance(v, list) and not isinstance(v[0], str) else v)
print("n_ref", len(st["zero_load_ref_per_episode"]), "keys", sorted(st.keys()))
EOF