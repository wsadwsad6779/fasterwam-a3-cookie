"""Bundle task7 evidence, upload to HF, verify sha256 by re-download, then prune
only regenerable Wan files (keep VAE). Token is read from a file and never printed."""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import time

os.environ['HF_ENDPOINT'] = 'https://huggingface.co'

P = '/workspace/fwv1/p_items'
PROJ = '/workspace/fwv1/repo/fasterwam_fingertip_force_v1'
WAN = '/workspace/fwv1/wan_base'
TOKEN_FILE = '/root/.cache/hf_token'
REPO = 'wsadwsad6779/fasterwam-a3-force-40k'
REMOTE = 't7_21k/evidence/t7_s1s2_evidence_20261123.tar.gz'
BUNDLE = '/workspace/fwv1/export/t7_s1s2_evidence.tar.gz'
UPLOAD_MANIFEST = P + '/GPU_UPLOAD_MANIFEST.json'
STORAGE = P + '/STORAGE_STATE.json'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def free_gb():
    s = os.statvfs('/workspace')
    return round(s.f_bavail * s.f_frsize / 1e9, 1)


def make_bundle():
    os.makedirs(os.path.dirname(BUNDLE), exist_ok=True)
    include = []
    include += sorted(glob.glob(P + '/*.json')) + sorted(glob.glob(P + '/*.md')) + sorted(glob.glob(P + '/*.csv'))
    include += sorted(glob.glob(P + '/videos/*.mp4')) + sorted(glob.glob(P + '/videos/trace_*.npz'))
    include += sorted(glob.glob(P + '/scripts/*.py')) + sorted(glob.glob(P + '/scripts/*.sh'))
    for name in ('TRAINING_RUN_STATE.json', 'CHECKPOINT_MANIFEST.json', 'EVAL_SEED_MANIFEST.json',
                 'VIDEO_MANIFEST.json', 'PROJECT_TASKS.json'):
        fp = os.path.join(PROJ, name)
        if os.path.exists(fp):
            include.append(fp)
    with tarfile.open(BUNDLE, 'w:gz') as tf:
        for fp in include:
            if os.path.getsize(fp) > 0:
                tf.add(fp, arcname=os.path.relpath(fp, '/workspace/fwv1'))
    return include


def upload_verify():
    from huggingface_hub import HfApi, hf_hub_download
    token = open(TOKEN_FILE).read().strip()
    api = HfApi(token=token)
    sha = sha256_file(BUNDLE)
    size = os.path.getsize(BUNDLE)
    info = {'repo': REPO, 'remote': REMOTE, 'bundle': BUNDLE, 'bytes': size, 'sha256': sha}
    try:
        info['whoami'] = api.whoami().get('name') if hasattr(api.whoami(), 'get') else None
    except Exception as ex:
        info['whoami_error'] = type(ex).__name__
    api.upload_file(path_or_fileobj=BUNDLE, path_in_repo=REMOTE, repo_id=REPO,
                    repo_type='model', commit_message='task7 S1-S2 evidence bundle')
    local = hf_hub_download(repo_id=REPO, filename=REMOTE, repo_type='model', token=token)
    got = sha256_file(local)
    info['verify_sha256'] = got
    info['verified'] = bool(got == sha)
    os.remove(local)
    return info


def prune():
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
        d = os.path.join(WAN, rel)
        if os.path.isdir(d):
            for root, _, files in os.walk(d):
                for f in files:
                    fp = os.path.join(root, f)
                    targets.append((fp, os.path.getsize(fp)))
    freed = 0
    removed = []
    for fp, sz in targets:
        try:
            os.remove(fp)
            freed += sz
            removed.append({'path': fp, 'bytes': sz})
        except Exception as ex:
            removed.append({'path': fp, 'error': type(ex).__name__})
    return removed, freed


def main():
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'free_gb_before': free_gb()}
    files = make_bundle()
    res['bundle_files'] = len(files)
    res['bundle_requested'] = [os.path.relpath(f, '/workspace/fwv1') for f in files]
    try:
        up = upload_verify()
        res['upload'] = up
    except Exception as ex:
        res['upload'] = {'error': type(ex).__name__ + ': ' + str(ex)[:300], 'verified': False}
    if res.get('upload', {}).get('verified'):
        removed, freed = prune()
        res['pruned'] = removed
        res['freed_gb'] = round(freed / 1e9, 2)
        res['prune_policy'] = ('only regenerable Wan repo files removed (3 DiT shards, T5 text encoder, index, '
                               'example assets); Wan2.2_VAE.pth and config.json kept; re-download method and '
                               'sha256 are in P4_WAN_FASTDL.json')
    res['free_gb_after'] = free_gb()
    json.dump(res, open(UPLOAD_MANIFEST, 'w'), indent=1)
    st = json.load(open(STORAGE)) if os.path.exists(STORAGE) else {}
    st.update({'checked_at': res['ts'], 'disk_free_gb': res['free_gb_after'],
               'status': 'ok' if res['free_gb_after'] >= 50 else 'low',
               'upload': {'status': 'verified' if res.get('upload', {}).get('verified') else 'failed',
                          'manifest': UPLOAD_MANIFEST,
                          'repo': REPO, 'remote': REMOTE,
                          'sha256': res.get('upload', {}).get('sha256')},
               'pruned_regenerable': [r['path'] for r in res.get('pruned', []) if 'bytes' in r]})
    json.dump(st, open(STORAGE, 'w'), indent=1)
    print(json.dumps({k: res.get(k) for k in ['free_gb_before', 'free_gb_after', 'bundle_files',
                                              'freed_gb', 'upload']}))


if __name__ == '__main__':
    main()