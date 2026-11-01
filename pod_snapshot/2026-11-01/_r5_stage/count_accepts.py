import json
import sys
from pathlib import Path

for d in sys.argv[1:]:
    rows = json.loads((Path(d) / "summary.json").read_text())
    n1 = [r for r in rows if len(r["clamps"]) == 1]
    ok = [r for r in n1 if r["clamps"][0]["accept"]]
    print(d, "rows=%d single_clamp=%d accept=%d" % (len(rows), len(n1), len(ok)))
    if rows:
        print("   settings_keys", sorted(rows[0]["settings"].keys()))
    for r in rows:
        st = r["settings"]
        print("   seed=%s var=%s band=%s clamps=%s stop=%s padmax=%.3f over2n=%d up=%d tilt=%d" % (
            r["seed"], st.get("variant", st.get("variant_index")), st.get("target_grip_force"),
            [bool(c["accept"]) for c in r["clamps"]], r["stop"], r["pad_max_n"],
            r["frames_over_2n"], r["moved_into_bin_upright"], r["moved_into_bin_tilted"]))
