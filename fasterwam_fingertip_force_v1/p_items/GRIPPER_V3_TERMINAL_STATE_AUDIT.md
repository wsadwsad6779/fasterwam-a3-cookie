# GRIPPER V3 TERMINAL STATE AUDIT

- episodes: 749
- first frame open (>=0.8): 749
- final frame open (>=0.8): 0
- final frame closed (<=0.6): 749
- final rise to open in last 40 frames: 0
- label last row open (>=0.8): 0
- verdict: **incomplete_terminal_release**

If final-open is 0, the recorded episodes end with the gripper still closed; the mid-episode release is present, but the final release after the last batch is not demonstrated in the data and must be handled before claiming a complete open/close cycle.