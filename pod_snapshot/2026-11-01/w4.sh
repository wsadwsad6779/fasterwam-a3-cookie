cat >> /workspace/_r5_stage/single_lowforce_probe.py <<'PY'


def _place_pose_single(self, clearance=0.0, column_index=None):
    # ordered_column() indexes a two-column demonstration order, which the
    # five-per-clamp experts only ever address with batch_index 0 or 1. Ten
    # single-cookie clamps drain the left source column first, so fill the
    # near target column for the first five and the far one for the last five.
    if column_index is None:
        order = getattr(self.env, "demonstration_column_order", (0, 1))
        column_index = order[0] if self.batch_index < 5 else order[1]
    return A3SameColumnBatchExpert._place_pose(self, clearance, column_index)


SingleLowForce._place_pose = _place_pose_single
PY
python3 -m py_compile /workspace/_r5_stage/single_lowforce_probe.py && echo COMPILE_OK
tail -3 /workspace/_r5_stage/single_lowforce_probe.py
echo DONE