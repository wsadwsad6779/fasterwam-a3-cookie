# F2_TRAINING_READINESS

- trainable params: action_expert 558881799 + proprio 36864 + film 534784 = 559453447
- video expert frozen; checkpoint estimate 5.59GB per milestone (7 -> 39.13GB)
- gripper-dim gradient (head row 6 norm): 0.7245 vs other rows mean 1.2432 (nonzero=True)
- class balance V2 train: close 0.9350 / open 0.0650 (mitigation: per-dim loss weight and balanced batch sampling, no label flip)
- force effectiveness: zero-init identity=True; after non-zero film, action L2(real vs zero)=1.5717, gripper sign flips=0.094
- inputs: 8D proprio, past-only 8-step force history with mask; no future/oracle inputs; text via cached T5 context
- datasets: V2 split 589/85/171 frozen test, quarantine 96 unknown identity; force normalization V2 train-only
- D3 decision: accept_training_data_with_ood_flag; D4: gripper verdict pass
- **F2_TRAINING_READY = true**