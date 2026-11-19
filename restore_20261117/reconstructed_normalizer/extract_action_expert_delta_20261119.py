"""Extract the action-expert delta from step_000100.pt and verify it bit-exactly.

The trial checkpoint is a full 24 GB state_dict; the only weights that differ from the
merged 40k base are the 602 `action_expert.*` tensors (563,140,103 params, ~1.13 GB bf16).
This writes a delta-only artefact and proves a reload is bit-identical. Optimizer state was
never saved, so step-200 continuation is explicitly impossible to do seamlessly.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CK = ROOT / 'model_repair_runs' / 'lora_300' / 'step_000100.pt'
OUTDIR = ROOT / 'model_repair_runs' / 'lora_300'
DELTA = OUTDIR / 'step_000100_action_expert_delta.pt'
MERGED = ROOT.parent / 'step_040000_merged.pt'


def sha(path, blk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(blk), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ck = torch.load(CK, map_location='cpu', mmap=True, weights_only=False)
    tr = list(ck['trainable'])
    model = ck['model']
    payload = {
        'delta': {k: model[k].clone() for k in tr},
        'trainable': tr,
        'step': int(ck.get('step', -1)),
        'base': str(MERGED),
        'base_step': 40000,
        'source_checkpoint': str(CK),
        'source_checkpoint_sha256': sha(CK),
        'optimizer_state': 'NEVER_SAVED',
        'resume_from_later_step': 'IMPOSSIBLE (no optimizer state; step_000200 corrupt)',
        'note': 'apply onto a model built from the merged base; all other modules are identical',
    }
    torch.save(payload, DELTA)
    d = torch.load(DELTA, map_location='cpu', mmap=True, weights_only=False)
    equal = sum(1 for k in tr if torch.equal(d['delta'][k], model[k]))
    out = {
        'delta_path': str(DELTA),
        'delta_bytes': DELTA.stat().st_size,
        'delta_sha256': sha(DELTA),
        'source_checkpoint_sha256': payload['source_checkpoint_sha256'],
        'trainable_tensors': len(tr),
        'tensors_bit_equal_on_reload': equal,
        'reload_verified': equal == len(tr),
        'optimizer_state_saved': False,
        'note': 'reconstruction = build model from step_040000_merged.pt, then '
                'model.load_state_dict(delta, strict=False)',
    }
    (OUTDIR / 'step_000100_action_expert_delta.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
