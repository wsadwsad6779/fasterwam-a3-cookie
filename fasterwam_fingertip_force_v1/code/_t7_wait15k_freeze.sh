#!/bin/bash
# Persistent waiter: after F0 15k lands + freeze, produce enriched F0_FROZEN_15K_FULL.json.
# Read-only; does not train; waits up to ~8h. Also writes WAIT15K_DONE.json marker.
set -u
P=/workspace/fwv1/p_items
RUN=/workspace/fwv1/f0_v3_baseline
EV=/workspace/fwv1/eval_fingertip_v1
PY=/opt/venv/bin/python
LOG=/workspace/fwv1/logs/wait15k_freeze.log
exec >> "$LOG" 2>&1
echo "WAIT15K_START $(date -u +%FT%TZ)"
for i in $(seq 1 480); do
  if [ -f "$P/F0_FROZEN_15K.json" ]; then echo "WAIT15K_FROZEN_SEEN"; break; fi
  if [ -f "$RUN/step_015000.pt" ] && [ -f "$EV/cmp_f0_15k/SUMMARY_step_15000.json" ]; then echo "WAIT15K_SUMMARY_SEEN"; break; fi
  sleep 60
done
while pgrep -f "[_]t7_train_f0.py" >/dev/null || pgrep -f "[_]t7_eval" >/dev/null; do sleep 15; done
CK15="$RUN/step_015000.pt"
SUM15="$EV/cmp_f0_15k/SUMMARY_step_15000.json"
S15=""; [ -f "$CK15" ] && S15=$(sha256sum "$CK15" | cut -d' ' -f1)
SH15=""; [ -f "$SUM15" ] && SH15=$(sha256sum "$SUM15" | cut -d' ' -f1)
$PY - ,"$CK15","$S15","$SH15","$SUM15" <<'PYEOF'
import json,sys,time,os
ck,s15,sh15,sum15=sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4]
metrics={"avg_in_target_strict":None,"exact_fill_rate":None,"n":None,"relaxed":"NOT_VERIFIED"}
if sum15 and os.path.exists(sum15):
    try:
        d=json.load(open(sum15)); metrics={"avg_in_target_strict":d.get("avg_in_target"),"exact_fill_rate":d.get("exact_fill_rate"),"n":len(d.get("results",[])),"relaxed":"NOT_VERIFIED (official _t7_eval has no relaxed)"}
    except Exception as e:
        metrics["parse_error"]=str(e)
out={"artifact":"F0_FROZEN_15K_FULL","ts":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
 "checkpoint":ck,"checkpoint_sha256":s15 or "NOT_VERIFIED",
 "summary15k":sum15,"summary15k_sha256":sh15 or "NOT_VERIFIED",
 "sha12k":"d3f2a3a8897dc346ac3523ab2499ef5134f35e1401d460b098603022bb1e60f4",
 "not_a_copy_of_12k": bool(s15 and not s15.startswith("d3f2a3a8")),
 "metrics":metrics,"eval_protocol":"r10/H32, g=2a-1, rot x5, exact_fill src==70, frozen V3.1, fail-closed",
 "hard_stop":15000,"fifteen_to_21k":"BLOCKED_USER"}
json.dump(out,open("/workspace/fwv1/p_items/F0_FROZEN_15K_FULL.json","w"),indent=1)
json.dump({"status":"WAIT15K_DONE","ts":out["ts"],"ckpt_sha":s15,"summary_sha":sh15},open("/workspace/fwv1/p_items/WAIT15K_DONE.json","w"),indent=1)
print(json.dumps(out))
PYEOF
echo "WAIT15K_DONE $(date -u +%FT%TZ)"
