#!/bin/bash
# After the F2 H-sweep finishes (H_SWEEP_RESULTS.json), verify the 9k package and prove 12k did NOT start,
# then SERIALLY run the H0 open-loop probe on F0 step_009000 (H=32/48/64, 0 training) -> H0_9K_PROBE.json.
# Inference only; does not train; does not touch F2 training state.
set -u
P=/workspace/fwv1/p_items
RUN=/workspace/fwv1/f0_v3_baseline
EV=/workspace/fwv1/eval_fingertip_v1
PY=/opt/venv/bin/python
LOG=/workspace/fwv1/logs/h0f0_after_hsweep.log
exec >> "$LOG" 2>&1
echo "H0F0_WAIT $(date -u +%FT%TZ)"
for i in $(seq 1 720); do [ -f "$P/H_SWEEP_RESULTS.json" ] && break; sleep 20; done
echo "H0F0_HSWEEP_DONE_SEEN $(date -u +%FT%TZ)"

# --- step 1: verify the 9k package + prove no 12k ---
CK="$RUN/step_009000.pt"
SUM=""; BYTES=""
if [ -f "$CK" ]; then SUM=$(sha256sum "$CK" | cut -d' ' -f1); BYTES=$(stat -c %s "$CK"); fi
HAS_SUM=no; [ -f "$EV/cmp_f0_9k/SUMMARY_step_9000.json" ] && HAS_SUM=yes
PG_CHAIN=$(pgrep -af '[_]t7_f0_continue_chain.sh' | tr '\n' ';' || true)
PG_TRAIN=$(pgrep -af '[_]t7_train_f0.py' | tr '\n' ';' || true)
PG_EVAL=$(pgrep -af '[_]t7_eval' | tr '\n' ';' || true)
$PY -c "
import json,sys,time
ck,summ,byt,has_sum,pgc,pgt,pge=sys.argv[1:8]
d={'artifact':'POST9K_9K_VERIFY','ts':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
'checkpoint':ck,'sha256':summ,'bytes':int(byt) if byt else None,
'summary_step_9000_exists':has_sum=='yes',
'procs':{'chain':pgc,'train':pgt,'eval':pge},
'no_12k_proof':(pgc.strip()=='' and pgt.strip()=='')}
json.dump(d,open('/workspace/fwv1/p_items/POST9K_9K_VERIFY.json','w'),indent=1)
print(json.dumps(d))
" "$CK" "$SUM" "$BYTES" "$HAS_SUM" "$PG_CHAIN" "$PG_TRAIN" "$PG_EVAL"
echo "H0F0_VERIFY_WRITTEN $(date -u +%FT%TZ)"

# --- step 2: ensure GPU idle, then H0 probe on F0 9k (open-loop, 0 training) ---
while pgrep -f "[_]t7_train_f0.py" >/dev/null || pgrep -f "[_]t7_eval" >/dev/null; do sleep 10; done
CKF0="$RUN/step_009000.pt"; [ -f "$CKF0" ] || CKF0=/root/fwv1_checkpoint_archive/f0_step_009000.pt
if [ ! -f "$CKF0" ]; then echo "H0F0_NO_CKPT"; exit 2; fi
echo "H0F0_RUN ckpt=$CKF0 $(date -u +%FT%TZ)"
H0_CKPT="$CKF0" H0_OUT="$P/H0_9K_PROBE.json" H0_NPZ="$P/H0_9K_CHUNKS.npz" \
H0_SEEDS=7500,7501,7502,7503,7504 H0_HORIZONS=32,48,64 \
  timeout 1800 $PY /workspace/_h0_zerotrain_probe.py
echo "H0F0_RC=$? $(date -u +%FT%TZ)"
echo "H0F0_DONE $(date -u +%FT%TZ)"
