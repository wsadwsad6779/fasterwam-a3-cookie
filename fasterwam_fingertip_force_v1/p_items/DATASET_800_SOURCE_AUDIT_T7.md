# DATASET_800_SOURCE_AUDIT_T7

Source: Eter0109/a3-cookie-multi-tasks (HF, LeRobot v2 layout); 845 episodes / 459177 frames.

## File hash verification vs FROZEN_DATA_MANIFEST.json

- data/chunk-000/file-000.parquet: match=True sha256=669b2eca42f4c5d4
- data/chunk-000/file-001.parquet: match=True sha256=717e444226e61821
- a3_episode_metadata.jsonl: match=True sha256=7d2fb566697dead9

## Metadata coverage
- metadata lines 845, unique ids 749, multi-source ids 96, parquet ids missing metadata 96

## Split (seed 20261123)
- train 675 / val 85 / test 85
- leakage check: {"disjoint": true, "complete": true, "counts": [675, 85, 85]}
- strata: {"local/cookie-2x10-complete-canonical-v4+local/cookie-2x5-complete-canonical-v4": 96, "local/cookie-2x5-complete-canonical-v4": 653, "unknown": 96}

## Contact coverage
- episodes with L pad contact: 845/845 (100.0%)
- frames with contact: 0.516
- R pad contact episodes: 0 (expected 0 in dataset)
- L pad max |force|: 2.16 N; R pad max: 0.00 N

## Existing provenance/dedup/diversity artifacts
- DATASET_800_SOURCE_LOCATION.md, DATASET_800_PROVENANCE.json, DATASET_800_DIVERSITY_REPORT.md, DATASET_800_DEDUPLICATION.csv, DATASET_800_FORCE_COVERAGE.json
- S2 re-verification: file hashes above; split isolation machine-checked