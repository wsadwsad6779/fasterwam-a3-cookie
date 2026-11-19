from pathlib import Path

import torch

BASE = Path('/workspace/eval_restore_20261117')
b = torch.load(BASE / 'step_040000_merged.pt', map_location='cpu', mmap=True,
               weights_only=False)
print('top', list(b.keys()), 'step', b.get('step'), 'dtype', b.get('torch_dtype'))
mot = b['mot']
print('mot nkeys', len(mot), 'head', list(mot.keys())[:6])
act = [k for k in mot if 'action' in k]
print('mot action keys', len(act), act[:5])
pe = b['proprio_encoder']
print('proprio nkeys', len(pe), list(pe.keys())[:4])

ck = torch.load(BASE / 'reconstructed_normalizer/model_repair_runs/lora_300/step_000100.pt',
                map_location='cpu', mmap=True, weights_only=False)
m = ck['model']
pairs = [('action_expert.blocks.11.ffn.0.weight', 'dit.mixtures.action.blocks.11.ffn.0.weight'),
         ('action_expert.blocks.19.self_attn.k.weight',
          'dit.mixtures.action.blocks.19.self_attn.k.weight')]
for a, c in pairs:
    if a in m and c in m:
        print('alias', a, '==', c, torch.equal(m[a], m[c]),
              'shape', tuple(m[a].shape))
print('trainable all action_expert:',
      all('action_expert' in k for k in ck['trainable']),
      'n', len(ck['trainable']))
