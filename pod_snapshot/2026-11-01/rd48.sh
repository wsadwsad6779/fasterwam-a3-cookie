pgrep -af replay_forces.py
tail -c 260 /workspace/demo30b.out
grep -c "###" /workspace/demo30b.out
grep -c accept /workspace/demo30b.out
