# GRIPPER V3 FORCE BY COMMAND LEVEL

- episodes: 749
- grasp (cmd<=0.35) steady: median pad force 1.371 N, fraction>0.2N 0.982
- release (0.35<cmd<0.7): median pad force 0.000 N, fraction>0.2N 0.000
- open (cmd>=0.8): median pad force 0.000 N
- right pad nonzero fraction: 0.00000
- verdict: **grasp_is_clamp**

Cookie grip-axis thickness is 0.0215 m and 2F-85 travel is 0.085 m, so a free
jaw at cmd 0.28 sits near 0.024 m; sustained pad force at the grasp level shows
the fingers are pressing the cookie (a real clamp), not hovering at a position.