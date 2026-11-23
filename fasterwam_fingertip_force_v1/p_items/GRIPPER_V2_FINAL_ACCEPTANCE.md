# GRIPPER_V2_FINAL_ACCEPTANCE

- episodes replayed under real MuJoCo control: 3, frames 1541
- label transitions confirmed within 16 steps: 11/11
- opening delay steps mean/median/max: 0.18181818181818182 / 0.0 / 1
- complete open->close->open->re-close cycles counted: 4
- closed frames 1435, open frames 106
- A3 raw action[7] vs 7D index 6 pm1 agreement: 1.0000
- phases per episode recorded in /workspace/fwv1/p_items/GRIPPER_PHASE_METRICS.json
- verdict: **pass**

## Notes
- raw action[7] is normalized opening (1 = open); 7D index 6 is pm1 (+1 close / -1 open); conversion uses <=0.6 / >=0.8 with deadband hold and min-hold 5, so no one-frame pulsing
- full 10-block task requires two release phases; cycle counts above confirm re-closing after release on multi-cycle episodes
- P5 traces cover the whole recorded episode; metrics-only episodes extend independence without new videos