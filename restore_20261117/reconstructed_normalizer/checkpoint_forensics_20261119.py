"""Checkpoint forensics v2: honest accounting of the bounded trial checkpoint.

Merged base layout: {'mot': 1438 keys, 'proprio_encoder': 2, 'step', 'torch_dtype',
'force_calibration'}. The trial checkpoint is a full model.state_dict() (24 GB, VAE + T5
included). Trainable selection was `'action_expert' in name`, i.e. no LoRA factors exist.
"""
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
CK = ROOT / 'model_repair_runs' / 'lora_300' / 'step_000100.pt'
MERGED = ROOT.parent / 'step_040000_merged.pt'
OUT = ROOT / 'c_class_eval_20261119' / 'checkpoint_forensics.json'


def base_lookup(b, key):
    if key.startswith('mot.'):
        k = key[4:]
        return b['mot'][k] if k in b['mot'] else None
    if key.startswith('action_expert.'):
        k = 'mixtures.action.' + key[len('action_expert.'):]
        return b['mot'][k] if k in b['mot'] else None
    if key.startswith('proprio_encoder.'):
        k = key[len('proprio_encoder.'):]
        return b['proprio_encoder'][k] if k in b['proprio_encoder'] else None
    return None


def main():
    ck = torch.load(CK, map_location='cpu', mmap=True, weights_only=False)
    m = ck['model']
    tr = list(ck['trainable'])
    base = torch.load(MERGED, map_location='cpu', mmap=True, weights_only=False)

    tot = sum(v.numel() for v in m.values() if hasattr(v, 'numel'))
    tr_n = sum(m[k].numel() for k in tr)
    prefixes = Counter(k.split('.')[0] for k in tr)
    aliases = [k for k in m if k.startswith('dit.mixtures.action.')]
    alias_equal = sum(1 for k in aliases[:40]
                      if torch.equal(m[k], m['action_expert.' + k[len('dit.mixtures.action.'):]]))

    tr_samples = {}
    for k in tr[:: max(1, len(tr) // 40)][:40]:
        ref = base_lookup(base, k)
        if ref is None:
            continue
        a, b = ref.float(), m[k].float()
        tr_samples[k] = {'shape': list(m[k].shape),
                         'max_abs_diff': float((a - b).abs().max()),
                         'rel': float((a - b).abs().max() / (a.abs().max() + 1e-12))}

    frozen = [k for k in m if k not in set(tr)]
    frozen_diff = {}
    checked = 0
    for k in frozen[:: max(1, len(frozen) // 60)][:60]:
        ref = base_lookup(base, k)
        if ref is None:
            continue
        checked += 1
        d = float((ref.float() - m[k].float()).abs().max())
        if d > 0:
            frozen_diff[k] = {'max_abs_diff': d,
                              'max_abs': float(ref.float().abs().max()),
                              'alias_of_action_branch': k.startswith('dit.mixtures.action.')}

    out = {
        'checkpoint': str(CK), 'step': int(ck.get('step', -1)),
        'merged_base': str(MERGED), 'merged_step': int(base.get('step', -1)),
        'checkpoint_keys': len(m), 'merged_keys': sum(len(v) for k, v in base.items()
                                                      if isinstance(v, dict)),
        'trainable_tensors': len(tr), 'trainable_prefixes': dict(prefixes),
        'trainable_all_action_expert': all(k.startswith('action_expert.') for k in tr),
        'trainable_params': int(tr_n), 'total_checkpoint_params': int(tot),
        'trainable_fraction_of_checkpoint': tr_n / tot,
        'delta_bytes_bf16': int(tr_n * 2),
        'lora_like_keys': [k for k in tr if 'lora' in k.lower()],
        'aliases_dit_mixtures_action': len(aliases),
        'aliases_bit_equal_checked': alias_equal,
        'alias_check_n': min(40, len(aliases)),
        'trainable_sample_vs_base': tr_samples,
        'frozen_keys_compared_to_base': checked,
        'frozen_nonzero_diff': frozen_diff,
        'verdict': 'FULL_ACTION_EXPERT_FINETUNE_NOT_LORA',
        'checkpoint_strategy_finding': 'step_000100.pt saves the entire model (VAE+T5), '
                                       '24 GB per snapshot; only the action_expert delta is '
                                       'needed and is ~delta_bytes_bf16',
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in (
        'trainable_tensors', 'trainable_all_action_expert', 'trainable_params',
        'total_checkpoint_params', 'trainable_fraction_of_checkpoint', 'delta_bytes_bf16',
        'lora_like_keys', 'aliases_bit_equal_checked', 'alias_check_n',
        'frozen_keys_compared_to_base', 'frozen_nonzero_diff', 'verdict')}, indent=2))


if __name__ == '__main__':
    main()
