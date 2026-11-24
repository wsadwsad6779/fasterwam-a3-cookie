# GRIPPER V3 ACCEPTANCE

- dim6 = continuous commanded opening (raw action[7], 1=open); V3 dim6 equals raw exactly (diff 0.0)
- real MuJoCo replay (ep 100/422/700): lag 0, RMSE 0.0039, corr 0.9997
- actual opening: open 0.967 / partial 0.463 / closed 0.287; closes then reopens then re-closes (1 full cycle per episode)
- training distribution: open>=0.8 5.2%, partial 43.3%, closed<=0.35 51.5% (V2 pm1 had 93.5% close)
- verdict: **pass** — the dataset opening is now represented and executable correctly
