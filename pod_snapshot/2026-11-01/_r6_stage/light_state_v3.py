'''v3 light state: persist exactly the tensors the optimiser owns.

v1 keyed the save set on the name pattern lora_ and silently dropped
proprio_encoder, which trainer.py:88 puts in the AdamW list. A v2 draft keyed on
requires_grad instead; on the live model that is 1637 tensors / 8,710,770,916
elements, because trainer.py:288-294 freezes by pattern without clearing the
flag on everything else -- writing it would have cost ~9 GB per save and filled
/workspace. Optimiser membership is the only set that means that training changed
this, so v3 selects by parameter identity and refuses to write when the
projected size exceeds the caller's budget.
'''
import json
import os
import shutil
from pathlib import Path

import torch

ADAPTER_FORMAT_V3 = 'fasterwam-lora-adapter-v3'
STATE_FORMAT_V3 = 'fasterwam-lora-light-state-v3'
V1_ADAPTER_FORMAT = 'fasterwam-lora-adapter-v1'

_WIDTH = {torch.bfloat16: 2, torch.float16: 2, torch.float32: 4, torch.float64: 8}


def _inner(optimizer):
    return getattr(optimizer, 'optimizer', optimizer)


def optimizer_membership(optimizer):
    inner = _inner(optimizer)
    members = [p for g in inner.param_groups for p in g['params']]
    return members


def select_saved(model, optimizer):
    '''Name -> tensor for every parameter the optimiser can modify.

    Identity first, because that is the truth. If the optimiser holds tensors
    that are not the model's own parameters (a sharding wrapper would), fall back
    to the trainer.py:85-88 construction order and accept it only when the
    shape/dtype sequence matches exactly.
    '''
    named = list(model.named_parameters())
    members = optimizer_membership(optimizer)
    ids = {id(p): n for n, p in named}
    hits = [(n, p) for p in members if id(p) in ids for n in [ids[id(p)]]]
    method = 'identity'
    if len(hits) != len(members):
        cand = [p for p in model.dit.parameters() if p.requires_grad]
        proprio = getattr(model, 'proprio_encoder', None)
        if proprio is not None:
            cand += [p for p in proprio.parameters() if p.requires_grad]
        cand_names = []
        for n, p in named:
            if id(p) in {id(c) for c in cand}:
                cand_names.append((n, p))
        sig = [(tuple(p.shape), str(p.dtype)) for p in cand]
        msig = [(tuple(p.shape), str(p.dtype)) for p in members]
        if sig != msig:
            raise RuntimeError('optimizer membership unresolved: identity matched %d of %d '
                               'and the trainer-order fallback does not reproduce it '
                               '(shapes %s vs %s). Refusing to guess what to save.'
                               % (len(hits), len(members), msig[:4], sig[:4]))
        hits, method = cand_names, 'positional_verified'
    seen = set()
    out = []
    for n, p in hits:
        if n in seen:
            continue
        seen.add(n)
        out.append((n, p))
    return {'selected': out, 'method': method, 'optimizer_members': len(members),
            'selected_tensors': len(out)}


def projected_bytes(selected):
    return sum(int(p.numel()) * _WIDTH.get(p.dtype, 4) for _, p in selected)


def free_bytes(path):
    d = Path(path)
    while not d.exists() and d != d.parent:
        d = d.parent
    return int(shutil.disk_usage(str(d)).free)


def coverage(model, optimizer):
    '''Three different sets, kept apart on purpose.'''
    named = list(model.named_parameters())
    sel = select_saved(model, optimizer)
    sel_names = {n for n, _ in sel['selected']}
    req = [n for n, p in named if p.requires_grad]
    lora = [n for n in req if 'lora_' in n]
    return {'named_tensors': len(named),
            'named_elements': sum(int(p.numel()) for _, p in named),
            'requires_grad_tensors': len(req),
            'requires_grad_elements': sum(int(p.numel()) for n, p in named if n in set(req)),
            'optimizer_members': sel['optimizer_members'],
            'saved_tensors': len(sel_names),
            'saved_elements': sum(int(p.numel()) for n, p in named if n in sel_names),
            'saved_lora_tensors': len([n for n in sel_names if 'lora_' in n]),
            'saved_nonlora_names': sorted(n for n in sel_names if 'lora_' not in n),
            'requires_grad_but_not_optimised': len([n for n in req if n not in sel_names]),
            'requires_grad_but_not_optimised_elements': sum(
                int(p.numel()) for n, p in named if p.requires_grad and n not in sel_names),
            'kept_by_v1': len(lora),
            'kept_by_v1_elements': sum(int(p.numel()) for n, p in named if n in set(lora)),
            'dropped_by_v1': sorted(n for n, p in named
                                    if p.requires_grad and 'lora_' not in n
                                    and n in sel_names),
            'dropped_by_v1_elements': sum(int(p.numel()) for n, p in named
                                          if p.requires_grad and 'lora_' not in n
                                          and n in sel_names),
            'selection_method': sel['method'],
            'projected_adapter_bytes': projected_bytes(sel['selected'])}


def save_adapter_v3(model, optimizer, path, meta=None, max_bytes=None):
    path = Path(path)
    cov = coverage(model, optimizer)
    sel = select_saved(model, optimizer)['selected']
    est = projected_bytes(sel)
    if max_bytes is None:
        max_bytes = int(free_bytes(path.parent) * 0.25)
    if est > max_bytes:
        raise RuntimeError('%s: v3 adapter would write %d B but the budget is %d B; '
                           'nothing was written.' % (path, est, max_bytes))
    payload = {'format': ADAPTER_FORMAT_V3,
               'lora': {n: p.detach().to('cpu', copy=True) for n, p in sel if 'lora_' in n},
               'extra_trainable': {n: p.detach().to('cpu', copy=True) for n, p in sel
                                   if 'lora_' not in n},
               'meta': dict(meta or {}, coverage=cov, projected_bytes=est)}
    tmp = path.with_name(path.name + '.tmp')
    torch.save(payload, tmp)
    actual = tmp.stat().st_size
    if actual > max_bytes:
        os.remove(tmp)
        raise RuntimeError('%s: wrote %d B, over budget %d B; removed.' % (path, actual, max_bytes))
    os.replace(tmp, path)
    return {'path': str(path), 'bytes': actual, 'coverage': cov}


def load_adapter_v3(model, path, map_location='cpu', allow_legacy=False):
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    fmt = payload.get('format')
    named = dict(model.named_parameters())
    if fmt == V1_ADAPTER_FORMAT:
        if not allow_legacy:
            raise RuntimeError('%s: v1 adapter holds only lora_ keys (%d tensors). Any '
                               'non-LoRA parameter the optimiser stepped is absent, so this '
                               'file cannot restore the trained model. Pass allow_legacy=True '
                               'to load the partial restore anyway.' % (path, len(payload['lora'])))
        blob = dict(payload['lora'])
    elif fmt == ADAPTER_FORMAT_V3:
        blob = dict(payload['lora'])
        blob.update(payload.get('extra_trainable', {}))
    else:
        raise ValueError('%s: not a recognised adapter checkpoint (%s)' % (path, fmt))
    unknown = sorted(set(blob) - set(named))
    if unknown:
        raise RuntimeError('%s: %d keys the model does not have, e.g. %s' % (path, len(unknown), unknown[:5]))
    bad = ['%s file%s model%s' % (n, list(blob[n].shape), list(named[n].shape))
           for n in blob if list(blob[n].shape) != list(named[n].shape)]
    if bad:
        raise RuntimeError('%s: shape mismatch: %s' % (path, bad[:5]))
    with torch.no_grad():
        for n, v in blob.items():
            p = named[n]
            p.copy_(v.to(device=p.device, dtype=p.dtype))
    return {'format': fmt, 'restored_tensors': len(blob),
            'restored_elements': sum(int(v.numel()) for v in blob.values()),
            'meta': payload.get('meta', {})}
