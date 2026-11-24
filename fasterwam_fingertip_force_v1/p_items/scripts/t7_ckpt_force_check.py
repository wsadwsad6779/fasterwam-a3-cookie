"""Inspect force-head presence/weights in a training checkpoint."""
import json
import os

import torch

P = os.environ.get('T7_CKPT_PATH', '/workspace/fwv1/train_runs/t7_smoke_force/step_000001.pt')
P2 = os.environ.get('T7_CKPT_PATH2', '')
OUT = os.environ.get('T7_CKPT_OUT', '/workspace/fwv1/p_items/T7_FORCE_HEAD_CKPT_CHECK.json')


def main():
    ck = torch.load(P, map_location='cpu', weights_only=False)
    fh = ck.get('force_head', {})
    n = sum(int(v.numel()) for v in fh.values())
    last = fh.get('3.weight')
    opt = ck.get('optimizer', {})
    out = {'path': P, 'step': int(ck.get('step', -1)),
           'keys': sorted(ck.keys()), 'force_head_keys': sorted(fh.keys()),
           'force_head_params': n,
           'last_layer_absmax': float(last.abs().max()) if last is not None else None,
           'last_layer_nonzero': bool((last != 0).any()) if last is not None else None,
           'optimizer_param_groups': len(opt.get('param_groups', [])),
           'optimizer_state_entries': len(opt.get('state', {})),
           'base_sha256': ck.get('base_sha256')}
    if P2 and os.path.exists(P2):
        ck2 = torch.load(P2, map_location='cpu', weights_only=False)
        fh2 = ck2.get('force_head', {})
        diffs = {k: float((fh[k].float() - fh2[k].float()).abs().max()) for k in fh if k in fh2}
        out['compare'] = {'path2': P2, 'step2': int(ck2.get('step', -1)),
                          'force_head_max_abs_diff': max(diffs.values()) if diffs else None,
                          'per_key': diffs}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()