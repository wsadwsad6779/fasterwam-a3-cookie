import hashlib
import json
import os
import time
import traceback

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

REPO = 'Eter0109/a3-cookie-multi-tasks'
D = '/workspace/fwv1/data/hf_videos'
P = '/workspace/fwv1/p_items'
LOG = '/workspace/fwv1/logs/video_download.log'
OUT = P + '/VIDEO_DOWNLOAD_STATE.json'
STATE = P + '/POD_JOB_STATE.json'
WANT_BYTES = 4512782439
WANT_FILES = 28
START = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(m):
    print(m, flush=True)
    with open(LOG, 'a') as f:
        f.write(m + '\n')


def st(status, note='', extra=None):
    d = {'started_at': START, 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'pid': os.getpid(), 'stage': 'video_download', 'log_path': LOG, 'status': status,
         'expected_artifacts': [D], 'next': 'real-batch training smoke with F2 force fusion'}
    if extra:
        d.update(extra)
    json.dump(d, open(STATE, 'w'), indent=1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def main():
    os.makedirs(D, exist_ok=True)
    st('running', note='hf-mirror snapshot of videos/** with backoff')
    log('START ' + START)
    ok = False
    for attempt in range(1, 9):
        try:
            snapshot_download(repo_id=REPO, repo_type='dataset', local_dir=D,
                              allow_patterns=['videos/**'], max_workers=2)
            ok = True
            log('DOWNLOAD_OK attempt=%d' % attempt)
            break
        except Exception as ex:
            log('RETRY %d %s: %s' % (attempt, type(ex).__name__, str(ex)[:200]))
            time.sleep(min(120, 20 * attempt))
    files = []
    for root, _, names in os.walk(D):
        for n in names:
            fp = os.path.join(root, n)
            files.append(fp)
    files.sort()
    recs = []
    total = 0
    for fp in files:
        sz = os.path.getsize(fp)
        total += sz
        recs.append({'path': os.path.relpath(fp, D), 'bytes': sz, 'sha256': sha256_file(fp)})
    res = {'worker': 'task7', 'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'repo': REPO, 'dir': D, 'download_ok': ok, 'files': len(recs),
           'bytes': total, 'want_bytes': WANT_BYTES, 'want_files': WANT_FILES,
           'matches_expected': bool(len(recs) == WANT_FILES and total == WANT_BYTES),
           'video_files': recs, 'log': LOG}
    json.dump(res, open(OUT, 'w'), indent=1)
    st('done' if (ok and res['matches_expected']) else 'partial',
       note='files=%d bytes=%d expected=%d/%d' % (len(recs), total, WANT_FILES, WANT_BYTES),
       extra={'files': len(recs), 'bytes': total})
    log('DONE ok=%s files=%d bytes=%d' % (ok, len(recs), total))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        log('TOPFAIL')
        log(traceback.format_exc()[:2000])
        st('failed', note=traceback.format_exc()[-500:])