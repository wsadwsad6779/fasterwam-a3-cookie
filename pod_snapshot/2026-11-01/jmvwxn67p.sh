cd /workspace/_r5_stage
python3 - <<EOF
import json, glob, os
carry_hi = []; post_hi = []; multi = 0
for f in sorted(glob.glob("/workspace/a3_eval/demos_*/summary.json")):
    try: rows = json.loads(open(f).read())
    except Exception as e:
        print("SKIP", f, e); continue
    for r in rows:
        cs = r.get("clamps") or []
        if len(cs) > 1: multi += 1
        for i, c in enumerate(cs):
            cp = c.get("carry_peak_pad_n") or 0.0
            pr = c.get("post_release_pad_n")
            if cp > 2.0: carry_hi.append((os.path.basename(os.path.dirname(f)), r["seed"], i, round(cp, 3), c.get("max_cookies_at_grip")))
            if pr is not None and pr > 0.05: post_hi.append((os.path.basename(os.path.dirname(f)), r["seed"], i, round(pr, 3)))
print("episodes_with_multiple_clamps", multi)
print("CARRY_GT_2N", len(carry_hi))
for x in carry_hi[:25]: print("  ", x)
print("POST_RELEASE_GT_0.05N", len(post_hi))
for x in post_hi[:25]: print("  ", x)
EOF