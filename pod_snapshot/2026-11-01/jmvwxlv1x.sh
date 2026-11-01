cd /workspace/_r5_stage
grep -n extras_released replay_wrongclosure.py
python3 - <<EOF
import json, numpy as np
r=json.loads(open("/workspace/a3_eval/demos_wrongclosure_v9/summary.json").read())
print(json.dumps(r[0]["retry_log"]))
z=np.load("/workspace/a3_eval/demos_wrongclosure_v9/expert_wrongclosure_7031.npz")
print(sorted(z.files))
ao=np.asarray(z["applied_opening"]); op=np.asarray(z["opening"]); at=np.asarray(z["cookies_at_grip"]) if "cookies_at_grip" in z.files else None
print("applied 90-118", [round(float(x),3) for x in ao[90:118]])
print("opening 90-118", [round(float(x),3) for x in op[90:118]])
if at is not None: print("at_grip 90-118", [int(x) for x in at[90:118]])
EOF