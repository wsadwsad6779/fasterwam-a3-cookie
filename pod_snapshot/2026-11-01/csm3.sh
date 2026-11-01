cd /workspace/_r5_stage
python3 - <<'PY'
from pathlib import Path
p = Path("replay_correct.py")
s = p.read_text(encoding="utf-8")
old = "{k: c[k] for k in"
new = "{k: c.get(k) for k in"
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new), encoding="utf-8")
print("patched")
PY
sha256sum replay_correct.py
python3 -m py_compile replay_correct.py && echo COMPILE_OK
python3 - <<'PY'
import json
rows = json.load(open("/workspace/a3_eval/demos_single_v30/summary.json"))
bad = [(i, r["seed"], r["pad_max_n"], r["frames_over_2n"], r["stop"][:44]) for i, r in enumerate(rows) if not r["clamps"][0]["accept"]]
print("variant seed padmax over2n stop")
for b in bad:
    print("  ", b)
PY
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
python3 -u /workspace/_r5_stage/replay_correct.py --seeds 7204 7228 --variants 3 27 --noise 0.0003 --retries 1 --max-steps 1600 --out /workspace/a3_eval/demos_correct_smoke2
date
echo __END__
