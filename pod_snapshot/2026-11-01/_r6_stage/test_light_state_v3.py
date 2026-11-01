'''Synthetic test for light_state_v3. CPU only, writes kilobytes.

Covers the two failure modes found on the real model: v1 saving by the lora_
name pattern (drops a stepped non-LoRA tensor) and a v2-style save by
requires_grad (would include a big frozen-but-flagged module). Also checks the
budget guard, the legacy refusal, and that an unresolvable optimiser is refused
rather than guessed.
'''
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, '/workspace/_r6_stage')
import light_state_v3 as V3

OUT = Path('/workspace/_r6_stage/out/v3test')
R = {}


class Toy(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.dit = nn.Module()
        self.dit.base = nn.Linear(8, 8)
        self.dit.q_lora = nn.Module()
        self.dit.q_lora.lora_A = nn.Parameter(torch.randn(4, 8) * 0.1)
        self.dit.q_lora.lora_B = nn.Parameter(torch.randn(8, 4) * 0.1)
        self.dit.video_expert = nn.Linear(64, 64)
        self.proprio_encoder = nn.Linear(8, 64)
        self.dit.base.requires_grad_(False)

    def optimisable(self):
        '''Exactly trainer.py:85-88: LoRA by pattern, plus the proprio encoder.'''
        lora = [p for n, p in self.named_parameters() if 'lora_' in n]
        return lora + [p for p in self.proprio_encoder.parameters() if p.requires_grad]


def build(seed=0):
    m = Toy(seed)
    opt = torch.optim.AdamW(m.optimisable(), lr=1e-3)
    return m, opt


def step(m, opt):
    for _ in range(5):
        opt.zero_grad()
        x = torch.randn(2, 8)
        (m.dit.q_lora.lora_B @ m.dit.q_lora.lora_A).sum().backward(retain_graph=True)
        m.proprio_encoder(x).sum().backward()
        opt.step()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m, opt = build()
    cov = V3.coverage(m, opt)
    R['coverage'] = cov
    R['check_decoy_requires_grad_but_not_optimised'] = (
        cov['requires_grad_but_not_optimised'] == 2)
    R['check_saved_excludes_decoy'] = all('video_expert' not in n for n, _ in
                                          V3.select_saved(m, opt)['selected'])
    R['check_proprio_selected'] = sorted(cov['saved_nonlora_names']) == [
        'proprio_encoder.bias', 'proprio_encoder.weight']

    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    step(m, opt)
    trained = {n: p.detach().clone() for n, p in m.named_parameters()}
    R['check_proprio_actually_moved'] = not torch.equal(before['proprio_encoder.weight'],
                                                        trained['proprio_encoder.weight'])
    R['check_frozen_base_untouched'] = torch.equal(before['dit.base.weight'],
                                                   trained['dit.base.weight'])

    v1_keys = sorted(n for n, _ in m.named_parameters() if 'lora_' in n)
    torch.save({'format': V3.V1_ADAPTER_FORMAT,
                'lora': {n: trained[n] for n in v1_keys}}, OUT / 'v1.pt')
    R['v1_keys'] = len(v1_keys)
    R['v1_drops'] = sorted(set(trained) - set(v1_keys) - {'dit.base.weight', 'dit.base.bias'})
    R['check_v1_drops_proprio'] = (set(cov['dropped_by_v1']) == {'proprio_encoder.bias',
                                                                  'proprio_encoder.weight'}
                                   and {'proprio_encoder.weight'} <= set(R['v1_drops']))

    saved = V3.save_adapter_v3(m, opt, OUT / 'v3.pt', meta={'global_step': 5})
    R['save'] = {'bytes': saved['bytes'], 'projected': cov['projected_adapter_bytes'],
                 'file_on_disk': (OUT / 'v3.pt').stat().st_size}
    R['check_projected_matches_actual'] = abs(saved['bytes'] - cov['projected_adapter_bytes']) < 5_000_000

    fresh, _ = build(seed=1)
    rep = V3.load_adapter_v3(fresh, OUT / 'v3.pt')
    fa = dict(fresh.named_parameters())
    eq, ne = [], []
    for n, _ in V3.select_saved(m, opt)['selected']:
        (eq if torch.equal(fa[n], trained[n]) else ne).append(n)
    R['load_v3'] = rep
    R['check_v3_bit_exact'] = (ne == [], eq)
    R['check_v3_kept_decoy_fresh'] = not torch.equal(fa['dit.video_expert.weight'],
                                                     trained['dit.video_expert.weight'])

    legacy_refused = None
    try:
        V3.load_adapter_v3(fresh, OUT / 'v1.pt')
    except RuntimeError as e:
        legacy_refused = str(e)[:160]
    R['v1_refused'] = legacy_refused
    fresh2, _ = build(seed=2)
    lr = V3.load_adapter_v3(fresh2, OUT / 'v1.pt', allow_legacy=True)
    f2 = dict(fresh2.named_parameters())
    R['legacy_restore'] = {'report': lr,
                           'proprio_still_random': not torch.equal(f2['proprio_encoder.weight'],
                                                                   trained['proprio_encoder.weight']),
                           'lora_restored_exact': torch.equal(f2['dit.q_lora.lora_B'],
                                                              trained['dit.q_lora.lora_B'])}

    try:
        V3.save_adapter_v3(m, opt, OUT / 'never.pt', max_bytes=10)
        R['budget_guard'] = 'DID NOT RAISE'
    except RuntimeError as e:
        R['budget_guard'] = str(e)[:160]
    R['check_budget_left_no_file'] = (not (OUT / 'never.pt').exists()
                                      and not (OUT / 'never.pt.tmp').exists())

    class Stub:
        param_groups = [{'params': [p.detach().clone() for p in m.optimisable()]}]

        @property
        def optimizer(self):
            return self
    try:
        V3.select_saved(m, Stub())
        R['unresolvable_optimizer'] = 'guessed silently'
    except RuntimeError as e:
        R['unresolvable_optimizer'] = str(e)[:160]

    (OUT / 'v3test.json').write_text(json.dumps(R, indent=2, default=str))
    print(json.dumps(R, indent=2, default=str)[:5200])


if __name__ == '__main__':
    main()
