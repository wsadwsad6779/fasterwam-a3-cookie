"""Separate two explanations for the check-3 adapter difference.

H1: LoRA `lora_A` is re-randomised per process, so run A and run B started from
    different adapters and the resume itself is fine.
H2: `load_light_state` did not apply the saved adapter, so B2 re-initialised.

The signature differs: under H1 `B_step5` and `B2_step10` agree to within five
Adam steps; under H2 they differ by a whole random draw.
"""
import itertools
from pathlib import Path

import torch

ROOT = Path("/workspace/a3_train_runs")
RUNS = {
    "A10": ROOT / "chk3_A/checkpoints/light/step_000010/adapter.pt",
    "B5": ROOT / "chk3_B/checkpoints/light/step_000005/adapter.pt",
    "B2_10": ROOT / "chk3_B2/checkpoints/light/step_000010/adapter.pt",
}

LORA = {}
for name, path in RUNS.items():
    payload = torch.load(path, map_location="cpu", weights_only=False)
    LORA[name] = payload["lora"]
    print(f"{name}: {path}")
    print(f"   keys={len(payload['lora'])} meta={payload.get('meta')}")

names = sorted(LORA["A10"])
for a, b in itertools.combinations(RUNS, 2):
    same_set = set(LORA[a]) == set(LORA[b]) == set(names)
    max_d, n_identical, worst = 0.0, 0, None
    for k in names:
        ta, tb = LORA[a][k], LORA[b][k]
        if torch.equal(ta, tb):
            n_identical += 1
            continue
        d = float((ta.float() - tb.float()).abs().max())
        if d > max_d:
            max_d, worst = d, k
    print(f"{a} vs {b}: keysets_equal={same_set} identical_tensors={n_identical}/{len(names)} "
          f"max|d|={max_d:.6g} worst={worst}")

# Split by role: lora_A holds the random draw, lora_B starts at zero.
for a, b in [("A10", "B5"), ("B5", "B2_10"), ("A10", "B2_10")]:
    for role in ("lora_A", "lora_B"):
        ks = [k for k in names if k.endswith(role)]
        d = max(float((LORA[a][k].float() - LORA[b][k].float()).abs().max()) for k in ks)
        print(f"{a} vs {b} {role}: n={len(ks)} max|d|={d:.6g}")

k0 = next(k for k in names if k.endswith("lora_A"))
print("sample", k0)
for name in RUNS:
    print(f"  {name}: {LORA[name][k0][:3, :4].tolist()}")
