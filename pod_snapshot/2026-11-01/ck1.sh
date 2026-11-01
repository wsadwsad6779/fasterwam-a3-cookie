python3 - <<'PYEOF'
import json
rows = json.load(open("/workspace/a3_eval/demos_correct_c1/summary.json"))
for r in rows:
    print("seed", r["seed"], "nclamps", len(r["clamps"]), "steps", r["steps"])
    for c in r["clamps"]:
        print("   keys", sorted(c.keys()))
        break
    break
PYEOF
echo __END__
