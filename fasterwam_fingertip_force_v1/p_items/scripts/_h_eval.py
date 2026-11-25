#!/usr/bin/env python3
"""Variable-H closed-loop eval wrapper. Reuses the deployed diagnostic evaluator
(_t7_eval_diag) but lets H (action_horizon) and replan equal T7_H, so it reports
relaxed + strict in-box at that horizon. Inference only; no training."""
import os
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
sys.path.insert(0, '/workspace')

import _t7_eval_diag as D

H = int(os.environ.get('T7_H', '32'))
D.ACTION_HORIZON = H   # module global used by infer_action + the execute loop
D.REPLAN = H           # execute the whole predicted chunk
print('H_EVAL_WRAPPER H=%d checkpoint=%s' % (H, os.environ.get('T7_CKPT')), flush=True)
D.main()
