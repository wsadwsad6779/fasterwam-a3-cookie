ls -la /workspace/*.out 2>/dev/null | tail -20
for f in /workspace/wc4.out /workspace/wc6fix.out /workspace/cc4.out /workspace/cc5.out; do echo -n "$f "; grep -c ^{ $f 2>/dev/null || echo NA; done