cd /workspace/_r5_stage
grep -n "OBSERVE_FRAMES\|RELEASE_OPEN\|hold-frames\|hold_frames" replay_wrongclosure.py
echo LAUNCHERS
ls -1 /workspace/_r5_stage/*.sh | tail -12
echo WC5ASH
cat run_wc5a.sh 2>/dev/null || ls run*