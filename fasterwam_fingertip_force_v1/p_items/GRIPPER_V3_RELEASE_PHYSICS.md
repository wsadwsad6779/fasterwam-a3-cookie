# GRIPPER V3 RELEASE PHYSICS AUDIT

- episodes: 749
- episodes with >=1 physical release (release plateau + jaw >=0.40): 749
- episodes ending released (final jaw >=0.40): 749
- release events total: 2996 (median per episode 4.0)
- jaw opening medians: grasp 0.311 / release 0.462 / open 0.973
- final command median 0.490, final jaw median 0.490
- command vs jaw: RMSE median 0.0234, corr median 0.99378
- verdict: **pass_physical_release**

The earlier terminal audit binarised opening at 0.8 and therefore counted the
0.49 release plateau as closed. With the release level taken from the data
(0.35<cmd<0.7) and the robot state as ground truth, every episode both
releases during the task and ends with the jaw physically open.