# FINGERTIP_FUSION_ARCHITECTURE

F0: original FasterWAM (8-d proprio, 7-d action).
F1: + ForceEncoder(4-ch fingertips, hist<=8) -> residual projection into action tokens.
F2: + ForceFiLM zero-init gamma/beta modulating action-expert hidden features (identity at init, params=527616).
F3: optional GRU history only if F1/F2 show gain.

Forbidden: chi16/17 as fingertips; future-force leakage; external gripper gating.
