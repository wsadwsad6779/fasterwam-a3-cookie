---
tags:
- robotics
- lerobot
- mujoco
- smolvla
size_categories:
- 10K<n<100K
---
# A3 front-close left-arm cookie transfer: 100 demonstrations

LeRobot v3.0 simulation dataset for the A3 cookie-transfer task. This is the left-only training projection of `a3_front_close_v2_100`, not a new independent collection.

- 100 episodes, 42,681 frames, 20 Hz control.
- Task: transfer 10 cookies into the target box using the same-column expert.
- State and action: 8 dimensions (`L_q1..L_q7,L_gripper`); stored actions are joint-position commands.
- Three embedded 256 x 256 RGB image streams: front, left wrist, right wrist.
- Auxiliary observations retain their original dual-arm dimensions; environment action dimension is 16.
- Front camera position: `(0.50, -0.18, 1.25)` m; workspace target: `(0.15, 0.23, 0.77)` m; vertical field of view: 48 degrees. Camera geometry is a prototype estimate, not a calibrated measurement.
- The train split contains all 100 episodes; no independent held-out evaluation split is provided.

`collection_summary.json` records collection configuration and source provenance. `a3_episode_metadata.jsonl` and `attempts.jsonl` retain collection metadata. Parquet files include image bytes.

Before upload, actual Parquet rows were verified to contain episodes 0 through 99, 42,681 total frames, finite 8-D states and actions, and all three embedded image streams.

Simulator: https://github.com/Eter0109/a3_dual_arm_sim

Camera defaults matching this dataset: https://github.com/Eter0109/a3_dual_arm_sim/commit/ff552c4fe54fe344aeaff4c65270ee0fd2af4153
