# SUPERSEDED by p_items/D3_FINAL_CONCLUSION.json (2026-11-24T01:05Z)

The 15-19x gap in this note came from open-loop replays that did not restore the recorded scene and, even with the exact scene, replay stored joint-position targets open-loop. The current closed-loop expert reproduces the dataset scale (1.45 vs 1.46-1.47 N). True dataset-vs-sim force OOD: none. See D3_FINAL_CONCLUSION.json.

---

# FINGERTIP_FORCE_DISTRIBUTION_AUDIT

- dataset (V2 train, 589 eps, 279473 frames): L pads mean 0.635/0.638 N, q99 1.46/1.47 N, max 1.76/1.83 N
- current MuJoCo replay of the SAME recorded actions (1541 frames): L pad max 26.40/35.42 N, q99 21.66/28.11 N, contact fraction 0.277/0.199
- ratio sim/dataset max: 15.0x / 19.3x
- V2 normalization: train-only mean/std (std floor 0.05), active mask (R pads inactive), clip +/-5, contact gate at 0.05 N, explicit OOD flag; no blind amplification
- OOD: 25.2% of current-sim frames exceed the train max on left pads; 11.9% saturate at clip

## Hypotheses
- units: same newtons, same channel order, no conversion bug
- timing: same sensor assembly after mj_step; no future leakage
- expert: replaying recorded actions still yields high forces -> not the expert policy
- remaining cause: dataset archive was recorded under a different sim/contact/sensor configuration or scaling; unrecoverable from published files
- consequence: train on dataset distribution; evaluation must report force OOD explicitly (no silent clipping)
- verdict: **accept_training_data_with_ood_flag**