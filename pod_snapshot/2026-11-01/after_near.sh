#!/bin/bash
LOG=/workspace/dagger_logs
while [ ! -f $LOG/near_all.done ]; do sleep 20; done
source /workspace/activate_fasterwam_a3.sh >/dev/null 2>&1
cd /workspace/fasterwam_a3_bridge/bridge
/opt/venv/bin/python - > $LOG/near_sel.log 2>&1 <<'PY'
import json, glob
rows = []
for p in sorted(glob.glob("outputs/dagger_near/seed*_s*/report.json")):
    r = json.load(open(p))
    rows.append(dict(seed=r["seed"], step=r["take_step"], recovered=bool(r.get("recovered")),
                     in_target=int(r.get("end_cookies_in_target", -1)), delivered=int(r.get("delivered", -1)),
                     phases=r.get("phase_counts"), steps=r.get("steps"), pad=r.get("max_pad_force"),
                     fail=r.get("failure_phase"), why=r.get("failure_reason"), replay=r.get("replay")))
for r in sorted(rows, key=lambda x: (x["seed"], x["step"])):
    print(json.dumps(r, ensure_ascii=False))
full = [f"{r["seed"]}:{r["step"]}" for r in rows if r["recovered"] and r["in_target"] >= 10]
open("/workspace/dagger_logs/near_takes.txt", "w").write(" ".join(full))
print("FULL_10:", full)
PY
TAKES=$(cat $LOG/near_takes.txt)
echo "takes=$TAKES"
if [ -n "$TAKES" ]; then
  python3 _next/dagger_to_dataset.py --takes $TAKES --out /workspace/a3_data/a3_fw_dagger_near > $LOG/near_convert.log 2>&1
  touch $LOG/near_convert.done
  /opt/venv/bin/python _next/check_a3_dagger_dataset.py /workspace/a3_data/a3_fw_dagger_near > $LOG/near_check.log 2>&1
  touch $LOG/near_check.done
fi
touch $LOG/near_pipeline.done
