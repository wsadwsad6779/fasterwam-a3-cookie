# TASK7 DELIVERABLES INDEX (Goal V2)

Generated 2026-11-23T19:02Z. Every row maps a required deliverable to its artifact and commit.
Status legend: DONE = verified artifact committed; RUNNING = autonomous job on pod; BLOCKED = cannot proceed without user action.

## S1 — base files / identity
| requirement | artifact | status |
|---|---|---|
| Wan ModelScope download + sha256 | p_items/P4_WAN_FASTDL.json | DONE (34.2GB, essential_bad=0) |
| official base identity (8D/7D/32/no force) | p_items/T7_BASE_IDENTITY.json | DONE (sha256 01be4efc..) |
| P4 full-model fwd/bwd smoke | p_items/P4_FULL_SMOKE.json | DONE (mot 0/0, loss 4.824, 24.28GB) |
| joint->7D verification | p_items/JOINT_TO_7D_VERIFICATION.json | DONE (pass, ep_start defect fixed) |
| Wan dependency proof | p_items/WAN_DEPENDENCY_CHECK.json | DONE |

## S2 / D2 — 845-episode acceptance, gripper, metadata
| requirement | artifact | status |
|---|---|---|
| DATASET_ACCEPTED decision | p_items/DATASET_845_ACCEPTED_MANIFEST.json | DONE (true; 749 accepted, 96 quarantine) |
| quality audit | p_items/DATASET_845_FINAL_QUALITY_AUDIT.md | DONE |
| provenance | p_items/DATASET_845_PROVENANCE.json + PROVENANCE_V2.json | DONE (HF rev 09a723a1, sim 08ebe8c) |
| physical dedup | p_items/DATASET_845_PHYSICAL_DEDUP.csv | DONE (845 unique groups, 0 action dup pairs) |
| layout/visual coverage | p_items/DATASET_845_LAYOUT_VISUAL_COVERAGE.json | DONE |
| train/val/test leakage | p_items/DATASET_845_TRAIN_VAL_TEST_LEAKAGE.json | DONE (group leakage 0) |
| frozen split | p_items/DATASET_845_SPLIT_MANIFEST_V2.json | DONE (589/85/171) |
| metadata reconciliation | p_items/DATASET_800_METADATA_RECONCILIATION.json | DONE |
| gripper frame-level analysis | p_items/GRIPPER_FRAME_ERROR_ANALYSIS.json/.md | DONE (97.66% true accuracy) |
| gripper V2 final acceptance | p_items/GRIPPER_V2_FINAL_ACCEPTANCE.md + GRIPPER_PHASE_METRICS.json | DONE (11/11, delay med 0, cycles) |
| replay videos | p_items/P5_REPLAY_VIDEO.json + 6 MP4 on pod | DONE |

## S3 / D3 / D5 — force + F2
| requirement | artifact | status |
|---|---|---|
| force distribution audit | p_items/FINGERTIP_FORCE_DISTRIBUTION_AUDIT.md | DONE (1.83N vs 35.4N = 15-19x) |
| normalization V2 (train-only) | p_items/FINGERTIP_FORCE_NORMALIZATION_V2.json | DONE |
| OOD acceptance | p_items/FORCE_OOD_ACCEPTANCE.json | DONE (accept_with_ood_flag, 25.2% OOD) |
| processor real-MuJoCo acceptance | p_items/FINGERTIP_FORCE_MUJOCO_ACCEPTANCE.json | DONE |
| F2 identity/grad/resume | p_items/F2_FUSION_FWD_BWD_TEST.json + F2_RESUME_TEST.json | DONE |
| real-batch F2 smoke | p_items/REAL_BATCH_F2_SMOKE.json | DONE |
| F2 trainable params/effectiveness/readiness | p_items/F2_TRAINABLE_PARAMETERS.json + F2_FORCE_EFFECTIVENESS_TEST.json + F2_TRAINING_READINESS.md | DONE (READY=true) |

## S4 — 21k bounded training + 3k milestones
| requirement | artifact | status |
|---|---|---|
| gate before training | p_items/T7_G4_GATE.json | DONE (PASSED_V2) |
| training driver + dataset | p_items/scripts/t7_train.py, t7_data.py | DONE (V2 split, AV1 decode, force V2) |
| functional resume | p_items/RESUME_FUNCTIONAL_TEST.json + RESUME_LOAD_VERIFY.json | DONE (loss diff 0.0; live ckpt strict restore) |
| training progress | p_items/T7_TRAINING_HEARTBEAT.json; last known step 1200 @18:30Z | RUNNING / MONITORING BLOCKED (pod 403) |
| checkpoints | /workspace/fwv1/train_runs/t7_21k/latest.pt (step 1200, 3.36GB); milestone files 3k-21k pending | RUNNING |
| milestone chain | tmux fwv1:chain3 (3k eval seeds 7100-7109 -> resume 6k ...) | RUNNING (last known) |
| watchdog | tmux fwv1:watchdog; logs /workspace/fwv1/logs/t7_watchdog.log | RUNNING (last known) |
| seed ledger | EVAL_SEED_MANIFEST.json (base 7001-7010 consumed; 7100-7709 milestones; 7900-7909 test) | DONE |
| degradation analysis | p_items/scripts/t7_degradation_analysis.py (finalizer runs at 21k) | READY |
| finalizer | p_items/scripts/t7_finalizer.sh (21k summary -> degradation -> S5) | ARMED |

## S5 — final independent test
| requirement | artifact | status |
|---|---|---|
| best-checkpoint selection | p_items/BEST_CHECKPOINT_SELECTION.json (produced by finalizer) | PENDING 21k |
| independent 10-seed bare-model test | p_items/FINAL_TEST_REPORT.md + VIDEO_MANIFEST_final.json | PENDING |
| force contribution ablation | p_items/FORCE_CONTRIBUTION.json | PENDING |

## Open blockers
- 2026-11-23T19:01Z: pod Jupyter API lost auth (cookies cleared, all endpoints 403 nginx). Training last seen alive step 1200 @18:30Z. User must sign in again; then run `p_items/scripts/t7_post_login_recover.sh` steps in JupyterLab Terminal.
- Large checkpoint cloud upload is blocked by network (HF direct unreachable, mirror read-only, GitHub per-file limits); all checkpoints kept on persistent /workspace and listed in CHECKPOINT_MANIFEST.json.