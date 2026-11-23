import glob, hashlib, json, os, tarfile, time

P = '/workspace/fwv1/p_items'
PROJ = '/workspace/fwv1/repo/fasterwam_fingertip_force_v1'
BUNDLE = '/workspace/fwv1/export/t7_s1s2_evidence.tar.gz'
INFO = P + '/EVIDENCE_BUNDLE_INFO.json'


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
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
    info = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'bundle': BUNDLE, 'bytes': os.path.getsize(BUNDLE), 'sha256': sha256_file(BUNDLE),
            'files': len(include)}
    json.dump(info, open(INFO, 'w'), indent=1)
    print(json.dumps(info))


if __name__ == '__main__':
    main()