# DATA_AUGMENTATION_DECISION

- episodes=100, windows=39481, unique_trajectories=100
- Case A: the corpus is 100 trajectories; any 800 count is windows/variants, not independent demos.
- Action: add genuinely independent successful demonstrations (layout/contact coverage first), not re-renders.
- Known safe ranges: x in [-10,10] mm, y in [-8,8] mm (anchor (0,0) bit-equal gate).
