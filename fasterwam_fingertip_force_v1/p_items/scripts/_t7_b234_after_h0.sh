#!/bin/bash
# Serial B2->B5->B3->B4 watcher, file-marker driven, single GPU mutex.
# Waits for H0_9K_PROBE.json, then runs each stage with a 25-min timeout; on rc!=0 or missing
# output writes <stage>_FAIL.json (rc + log tail) and CONTINUES. Never blocks F0/12k rules.
set -u
P=/workspace/fwv1/p_items
PY=/opt/venv/bin/python
LOG=/workspace/fwv1/logs/b234_after_h0.log
exec >> "$LOG" 2>&1
echo "B234_WAIT $(date -u +%FT%TZ)"
for i in $(seq 1 900); do [ -f "$P/H0_9K_PROBE.json" ] && break; sleep 20; done
echo "B234_H0_SEEN $(date -u +%FT%TZ)"
# single-GPU mutex: wait until no heavy python job is running
while pgrep -f "[_]t7_train_f0.py" >/dev/null || pgrep -f "[_]t7_eval" >/dev/null \
   || pgrep -f "[_]h_eval.py" >/dev/null || pgrep -f "[_]h0_zerotrain_probe.py" >/dev/null; do sleep 10; done
echo "B234_GPU_IDLE $(date -u +%FT%TZ)"

run_stage(){
  name="$1"; script="$2"; out="$3"
  echo "B234_RUN $name $(date -u +%FT%TZ)"
  slog="/workspace/fwv1/logs/b234_$name.log"
  timeout 1500 "$PY" "$script" > "$slog" 2>&1
  rc=$?
  echo "B234_RC $name $rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ] || [ ! -f "$out" ]; then
    "$PY" -c "
import json,os,time
lg='$slog'
t=open(lg).read()[-900:] if os.path.exists(lg) else ''
json.dump({'stage':'$name','rc':$rc,'log_tail':t,'ts':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())},open('$P/${name}_FAIL.json','w'),indent=1)
"
    echo "B234_FAIL_WRITTEN $name $(date -u +%FT%TZ)"
  fi
}

run_stage B2 /workspace/_b2_sixphase_teacher.py    "$P/B2_SIXPHASE_TEACHER.json"
run_stage B5 /workspace/_b5_lgrip_lforce_gradcos.py "$P/B5_LGRIP_LFORCE_GRADCOS.json"
run_stage B3 /workspace/_b3_force_ablation_expand.py "$P/B3_FORCE_ABLATION_EXPANDED.json"
run_stage B4 /workspace/_b4_state_clone.py          "$P/B4_STATE_CLONE_043_028.json"
echo "B234_DONE $(date -u +%FT%TZ)"
