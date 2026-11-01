#!/usr/bin/env bash
source /workspace/activate_fasterwam_a3.sh >/dev/null 2>&1
cd /workspace/fasterwam_a3_bridge/bridge
mkdir -p /workspace/dagger_logs
run() {
  A3_RESUME_HELD="$4" A3_RESUME_PHASE="$3" python3 _next/dagger_resume.py --seed "$1" --take "$2" --dump-frames --no-video --expert-steps 900 --out outputs/dagger_resume > "/workspace/dagger_logs/resume_$5.log" 2>&1
}
run 11 506 close "2 3 4 5 7" 11_506 &
run 11 530 close "2 3 4 5 7" 11_530 &
run 21 190 close "1 2 3 4 5" 21_190 &
run 21 226 lift "0 1 2 3 4 5" 21_226 &
wait
touch /workspace/dagger_logs/resume_all.done
