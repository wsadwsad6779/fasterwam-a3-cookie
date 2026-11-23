# GRIPPER_LABEL_V2_FINAL_AUDIT

Real MuJoCo replay of recorded expert actions (left arm + gripper).

## episode 0
- replay steps: 466 / dataset frames 466
- label close frac: 0.9334763948497854, actual opening mean(close)=0.37149051413924766, mean(open)=0.9077054540740475
- best lag: {"lag": 0, "acc": 0.853689284390063, "close_actual_closed_frac": 0.9977011494252873, "open_actual_open_frac": 0.7096774193548387}
- transitions confirmed: 3/3, pulses=0
- pad force max (in/out): [26.40262794494629, 10.668743133544922]
- videos: ['/workspace/fwv1/p_items/videos/P5_replay_ep0000_front.mp4', '/workspace/fwv1/p_items/videos/P5_replay_ep0000_left_wrist.mp4']

## episode 422
- replay steps: 475 / dataset frames 475
- label close frac: 0.9347368421052632, actual opening mean(close)=0.38686980764743023, mean(open)=0.9077783074063366
- best lag: {"lag": 0, "acc": 0.8537125835512933, "close_actual_closed_frac": 0.9977477477477478, "open_actual_open_frac": 0.7096774193548387}
- transitions confirmed: 3/3, pulses=0
- pad force max (in/out): [23.652847290039062, 35.41891098022461]
- videos: ['/workspace/fwv1/p_items/videos/P5_replay_ep0422_front.mp4', '/workspace/fwv1/p_items/videos/P5_replay_ep0422_left_wrist.mp4']

## episode 844
- replay steps: 600 / dataset frames 1063
- label close frac: 0.9266666666666666, actual opening mean(close)=0.3735177789031471, mean(open)=0.9046848984016175
- best lag: {"lag": 0, "acc": 0.8600392413342054, "close_actual_closed_frac": 0.9928057553956835, "open_actual_open_frac": 0.7272727272727273}
- transitions confirmed: 5/5, pulses=0
- pad force max (in/out): [21.29750633239746, 30.381895065307617]
- videos: ['/workspace/fwv1/p_items/videos/P5_replay_ep0844_front.mp4', '/workspace/fwv1/p_items/videos/P5_replay_ep0844_left_wrist.mp4']

Verdict: pass