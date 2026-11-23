"""Restore task7 state on a fresh pod: download artifacts listed in GPU_UPLOAD_MANIFEST.json,
verify sha256, then print the exact resume command. No secrets are stored here."""
import json
import os
import sys

import requests

MANIFEST = os.environ.get('GPU_UPLOAD_MANIFEST', '/workspace/fwv1/p_items/GPU_UPLOAD_MANIFEST.json')
ROOT = '/workspace/fwv1'


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
    man = json.load(open(MANIFEST))
    uploads = man.get('uploads') or []
    if not uploads:
        print('manifest has no uploads yet; nothing to restore')
        return 0
    bad = 0
    for u in uploads:
        url = u['url']
        dst = os.path.join(ROOT, u['rel'])
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        print('GET', url, '->', dst)
        with requests.get(url, stream=True, timeout=3600) as r:
            r.raise_for_status()
            with open(dst, 'wb') as f:
                for chunk in r.iter_content(1 << 22):
                    f.write(chunk)
        got = sha256_file(dst)
        ok = got == u['sha256']
        print('sha256', 'OK' if ok else 'MISMATCH', got)
        if not ok:
            bad += 1
    print('restore resume:', man.get('resume_command', 'see TRAINING_RUN_STATE.json'))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())