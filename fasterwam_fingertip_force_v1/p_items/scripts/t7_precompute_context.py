import glob
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, '/workspace/fwv1/repo/stage_40k/fw/src')
import torch
from fasterwam.models.wan22.helpers.loader import _load_registered_model
from fasterwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

WAN = '/workspace/fwv1/wan_base'
T5 = WAN + '/models_t5_umt5-xxl-enc-bf16.pth'
WANT_BYTES = 11361920418
WANT_SHA = '7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'
CACHE = '/workspace/fwv1/data/text_embeds_cache'
OUT = '/workspace/fwv1/p_items/CONTEXT_CACHE_STATE.json'
PROMPT = "A video recorded from a robot's point of view executing the following instruction: transfer 10 cookies into target box"
CONTEXT_LEN = 128


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'context_len': CONTEXT_LEN}
    size = os.path.getsize(T5)
    res['t5_bytes'] = size
    res['t5_size_ok'] = size == WANT_BYTES
    assert res['t5_size_ok'], (size, WANT_BYTES)
    res['t5_sha256'] = sha256_file(T5)
    res['t5_sha_ok'] = res['t5_sha256'] == WANT_SHA
    assert res['t5_sha_ok'], res['t5_sha256']
    tok_files = []
    for root, _dirs, names in os.walk(WAN):
        if 'spiece.model' in names:
            tok_files.append(os.path.join(root, 'spiece.model'))
    assert tok_files, 'tokenizer dir not found'
    tok_dir = os.path.dirname(tok_files[0])
    res['tokenizer_found'] = True
    enc = _load_registered_model(T5, 'wan_video_text_encoder', torch.bfloat16, 'cuda')
    tok = HuggingfaceTokenizer(name=tok_dir, seq_len=CONTEXT_LEN, clean='whitespace')
    ids, mask = tok(PROMPT, return_mask=True, add_special_tokens=True)
    ids = ids.to('cuda')
    mask = mask.to('cuda', dtype=torch.bool)
    with torch.no_grad():
        emb = enc(ids, mask)
    context = emb[0].float().cpu()
    mask2 = mask[0].cpu()
    res['context_shape'] = list(context.shape)
    res['mask_true'] = int(mask2.sum())
    res['context_finite'] = bool(torch.isfinite(context).all())
    os.makedirs(CACHE, exist_ok=True)
    h = hashlib.sha256(PROMPT.encode('utf-8')).hexdigest()
    path = os.path.join(CACHE, '%s.t5_len%d.wan22ti2v5b.pt' % (h, CONTEXT_LEN))
    torch.save({'context': context, 'mask': mask2}, path)
    res['cache_path'] = path
    res['cache_sha256'] = sha256_file(path)
    res['cache_bytes'] = os.path.getsize(path)
    res['verdict'] = 'pass' if (res['t5_sha_ok'] and res['context_finite']
                                and res['context_shape'] == [CONTEXT_LEN, 4096]) else 'fail'
    json.dump(res, open(OUT, 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ['verdict', 't5_size_ok', 't5_sha_ok', 'context_shape',
                                          'mask_true', 'cache_bytes', 'context_finite']}))


if __name__ == '__main__':
    main()