# GRIPPER_FRAME_ERROR_ANALYSIS

Thresholds: closed=0.70, open=0.85.

| episode | frames | close acc | open acc | frame acc | steady acc | open-err mean | near-trans err | steady err |
|---|---|---|---|---|---|---|---|---|
| trace_ep0000.npz | 466 | 0.9977 | 0.7097 | 0.9785 | 1.0000 | 0.715 | 10 | 0 |
| trace_ep0422.npz | 475 | 0.9977 | 0.7097 | 0.9789 | 1.0000 | 0.715 | 10 | 0 |
| trace_ep0844.npz | 600 | 0.9928 | 0.7273 | 0.9733 | 1.0000 | 0.701 | 16 | 0 |

Aggregate: frames=1541, errors=36, frame accuracy=0.9766
- errors near transition (<=15 steps): 36 (100.00%)
- steady-state errors: 0 (0.00%)
- label-close errors: 6; label-open errors: 30

Frame accuracy 0.9766. Almost all errors are label-OPEN frames whose actual opening settles in [0.75, 0.85) after release rather than reaching the strict 0.85 fully-open threshold; label-CLOSE frames are essentially always satisfied. This is a threshold/settling effect, not a wrong-direction or jitter failure: every label transition was confirmed and there were zero pulses.