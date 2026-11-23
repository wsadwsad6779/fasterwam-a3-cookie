# DATASET_845_FINAL_QUALITY_AUDIT

- dataset_accepted: **True**
- episodes: 845 (accepted known=749, quarantine unknown identity=96, rejected=0)
- task class resolution: metadata single=653, metadata frame-match=96, unknown=96
- byte-level action duplicate pairs: 0
- physical groups: 845; group leakage before=0 after=0
- split v2: {"train": 589, "val": 85, "test": 171}
- layout coverage: source_column={"1": 749, "None": 96} task_class={"2x5": 749, "2x10_inferred": 96} unknown_identity=96
- 2x5 and 2x10 are never mixed in training; 96 unknown-identity episodes are frozen test-only
- files: PROVENANCE/DEDUP/LAYOUT/LEAKAGE/SPLIT_V2 + this manifest