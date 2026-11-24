# MILESTONE REPORT base / 3k / 6k

- base: exact_fill 0/10, max in_target 0, grasps mean 6.2, close_cmd_frac 0.949, safety 0, pad max 283.45, steps mean 600
- 3k: exact_fill 0/10, max in_target 0, grasps mean 0, close_cmd_frac 0.983, safety 0, pad max 188.77, steps mean 600
- 6k: exact_fill 0/10, max in_target 0, grasps mean 0, close_cmd_frac 0.979, safety 0, pad max 0.0, steps mean 600

3k/6k checkpoints give 0 exact_fill and 0 blocks in target; gripper closed ~96-98% of steps, grasps occur but no successful placement; zero safety stops

action loss is low but approach-open/release behavior is not learned yet; milestones 9k-21k carry the real signal; anchor protocol unchanged.