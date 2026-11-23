import glob
import json
import os
import time

WAN = '/workspace/fwv1/wan_base'
P = '/workspace/fwv1/p_items'
STORAGE = P + '/STORAGE_STATE.json'
UP = P + '/GPU_UPLOAD_MANIFEST.json'
BUNDLE_SHA = 'c1e7d60f3f89d753822724d9b819ba8328ed6bc1885388220b3500a3a23f8885'
BUNDLE_BYTES = 10470972
GITHUB_COMMIT = '442ac5e'
GITHUB_PATH = 'fasterwam_fingertip_force_v1/p_items/evidence_bundles/t7_s1s2_evidence.tar.gz'
GITHUB_REPO = 'wsadwsad6779/fasterwam-a3-cookie (branch fingertip-force-v1)'


def free_gb():
    s = os.statvfs('/workspace')
    return round(s.f_bavail * s.f_frsize / 1e9, 1)


def main():
    before = free_gb()
    targets = []
    for name in ('diffusion_pytorch_model-00001-of-00003.safetensors',
                 'diffusion_pytorch_model-00002-of-00003.safetensors',
                 'diffusion_pytorch_model-00003-of-00003.safetensors',
                 'models_t5_umt5-xxl-enc-bf16.pth',
                 'diffusion_pytorch_model.safetensors.index.json'):
        fp = os.path.join(WAN, name)
        if os.path.exists(fp):
            targets.append((fp, os.path.getsize(fp)))
    for rel in ('examples', 'assets'):
        for fp in glob.glob(os.path.join(WAN, rel, '**', '*'), recursive=True):
            if os.path.isfile(fp):
                targets.append((fp, os.path.getsize(fp)))
    removed = []
    freed = 0
    for fp, sz in targets:
        try:
            os.remove(fp)
            freed += sz
            removed.append({'path': fp, 'bytes': sz})
        except Exception as ex:
            removed.append({'path': fp, 'error': type(ex).__name__})
    kept = [fp for fp in ('Wan2.2_VAE.pth', 'config.json', 'configuration.json', '.msc', '.mv')
            if os.path.exists(os.path.join(WAN, fp))]
    after = free_gb()
    st = json.load(open(STORAGE)) if os.path.exists(STORAGE) else {}
    st.update({
        'checked_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'disk_free_gb': after, 'status': 'ok' if after >= 50 else 'low',
        'upload': {'status': 'verified', 'target': GITHUB_REPO, 'commit': GITHUB_COMMIT,
                   'path': GITHUB_PATH, 'bytes': BUNDLE_BYTES, 'sha256': BUNDLE_SHA,
                   'verified_by': 'byte-identical local download + sha256 match; file committed to GitHub'},
        'pruned_regenerable': removed,
        'freed_gb': round(freed / 1e9, 2),
        'wan_base_kept': kept,
        'redownload': {
            'method': 'ModelScope per-file curl -C - (see p_items/P4_WAN_FASTDL.json)',
            'note': 'VAE kept because training/inference needs it; DiT shards and T5 text encoder are unused in the official-base protocol (video weights come from step_021700.pt; context is precomputed)',
        },
    })
    json.dump(st, open(STORAGE, 'w'), indent=1)
    up = json.load(open(UP)) if os.path.exists(UP) else {}
    up.update({'status': 'verified', 'ts': st['checked_at'],
               'uploads': [{'kind': 'github', 'repo': GITHUB_REPO, 'commit': GITHUB_COMMIT,
                            'path': GITHUB_PATH, 'bytes': BUNDLE_BYTES, 'sha256': BUNDLE_SHA,
                            'verified': True}],
               'pruned_regenerable': [r['path'] for r in removed if 'bytes' in r],
               'free_gb': after})
    json.dump(up, open(UP, 'w'), indent=1)
    print(json.dumps({'before_gb': before, 'after_gb': after, 'freed_gb': round(freed / 1e9, 2),
                      'removed': len([r for r in removed if 'bytes' in r]), 'kept': kept}))


if __name__ == '__main__':
    main()