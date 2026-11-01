for f in wc4 wc6fix cc4 cc5; do echo -n "$f "; grep -c ^{ /workspace/_r5_stage/${f}.out 2>/dev/null || echo NA; done
ps -eo pid,etimes,cmd | grep -E "replay_(correct|wrongclosure)" | grep -v grep