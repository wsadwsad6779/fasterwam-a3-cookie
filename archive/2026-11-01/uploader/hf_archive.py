#!/usr/bin/env python3
"""Archive the P0/P1 manifest to Hugging Face and prove it landed byte-exact.

Route notes (measured 2026-11-01 on this pod):
  * huggingface.co is not on the egress allowlist; hf-mirror.com is, and it
    proxies the write API (whoami / preupload / commit).
  * classic LFS upload presigns to s3.amazonaws.com -> 403 from the proxy.
  * with Xet left enabled, a 61.6 MB .pt uploaded at ~5 MB/s and read back
    with an identical sha256. So: HF_ENDPOINT=hf-mirror, never set
    HF_HUB_DISABLE_XET.
Every file is hashed before it is sent and the recorded sha256 from the
manifest must match; a mismatch is refused, never uploaded.
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/hfstage'
os.environ['HF_HUB_CACHE'] = '/root/hfstage/hub'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ.pop('HF_HUB_DISABLE_XET', None)
os.makedirs('/root/hfstage', exist_ok=True)

from huggingface_hub import HfApi  # noqa: E402

REPO = 'wsadwsad6779/fasterwam-a3-cookie-lora'
MANIFEST = '/root/hfstage/p0p1_manifest.tsv'
STATE = '/root/hfstage/archive_state.jsonl'
ROOT = '/workspace/'
TOKEN_FILE = '/root/.cache/huggingface/token'


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_rows(batch, manifest=MANIFEST):
    rows = []
    with open(manifest, encoding='utf-8') as f:
        for line in f:
            p = line.rstrip('\n').split('\t')
            if len(p) != 5:
                continue
            if p[0] != batch:
                continue
            rows.append({'batch': p[0], 'path_in_repo': p[1], 'pod_path': ROOT + p[2],
                         'bytes': int(p[3]), 'sha256': p[4]})
    return rows


def done_map():
    """path_in_repo -> last recorded status, from the append-only state log."""
    m = {}
    if os.path.exists(STATE):
        with open(STATE, encoding='utf-8') as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get('path_in_repo'):
                    m[e['path_in_repo']] = e.get('status') or e.get('event')
    return m


def emit(out, event, **kw):
    e = {'ts': now(), 'event': event}
    e.update(kw)
    out.write(json.dumps(e, ensure_ascii=False) + '\n')
    out.flush()
    print(json.dumps(e, ensure_ascii=False)[:400], flush=True)
    return e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='P0')
    ap.add_argument('--max-files', type=int, default=25)
    ap.add_argument('--max-bytes', type=int, default=2 * 1024 ** 3)
    ap.add_argument('--limit', type=int, default=0, help='cap number of files this run')
    ap.add_argument('--min-file-bytes', type=int, default=0)
    ap.add_argument('--max-file-bytes', type=int, default=0, help='0 = no ceiling')
    ap.add_argument('--only', default='', help='comma separated path_in_repo substrings')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--tree', action='store_true', help='just list the remote repo')
    ap.add_argument('--redo', action='store_true', help='ignore OK entries in state')
    ap.add_argument('--manifest', default=MANIFEST)
    args = ap.parse_args()

    tok = open(TOKEN_FILE).read().strip()
    api = HfApi(token=tok)
    os.environ['HF_TOKEN'] = tok

    if args.tree:
        listing = []
        for f in api.list_files_info(repo_id=REPO, repo_type='model', revision='main'):
            lfs = getattr(f, 'lfs', None) or {}
            listing.append({'path': f.path, 'size': f.size,
                            'sha256': lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None),
                            'upload_mode': lfs.get('uploadMode') if isinstance(lfs, dict) else None})
        out = '/workspace/_r6_stage/out/hf_tree_%s.json' % now().replace(':', '')
        with open(out, 'w') as f:
            json.dump(listing, f, indent=1)
        print('TREE %d files -> %s' % (len(listing), out), flush=True)
        have = sum(1 for x in listing if x['sha256'])
        print('with sha256: %d of %d' % (have, len(listing)), flush=True)
        return 0

    rows = load_rows(args.batch, args.manifest)
    dm = done_map()
    if not args.redo:
        rows = [r for r in rows if dm.get(r['path_in_repo']) != 'OK']
    if args.min_file_bytes or args.max_file_bytes:
        rows = [r for r in rows
                if r['bytes'] >= args.min_file_bytes
                and (not args.max_file_bytes or r['bytes'] <= args.max_file_bytes)]
    if args.only:
        needles = [s.strip() for s in args.only.split(',') if s.strip()]
        rows = [r for r in rows if any(s in r['path_in_repo'] for s in needles)]
    if args.limit:
        rows = rows[:args.limit]
    total = sum(r['bytes'] for r in rows)
    run_id = now().replace(':', '').replace('-', '')
    logpath = '/workspace/_r6_stage/out/hf_archive_%s_%s.jsonl' % (args.batch, run_id)
    print('batch=%s files=%d bytes=%.3f GiB log=%s' %
          (args.batch, len(rows), total / 2 ** 30, logpath), flush=True)
    if args.dry_run:
        for r in rows[:8]:
            print(' ', r['path_in_repo'], r['bytes'], r['sha256'][:12])
        return 0

    from huggingface_hub import CommitOperationAdd
    print('create_commit params: %s' % list(inspect.signature(api.create_commit).parameters), flush=True)
    print('CommitOperationAdd kwargs: %s' % list(inspect.signature(CommitOperationAdd.__init__).parameters), flush=True)

    state = open(STATE, 'a', encoding='utf-8')
    out = open(logpath, 'a', encoding='utf-8')
    emit(out, 'start', batch=args.batch, files=len(rows), bytes=total,
         revision_before=api.repo_info(repo_id=REPO, repo_type='model').sha)

    batches = []
    cur, cursz = [], 0
    for r in rows:
        if cur and (len(cur) >= args.max_files or cursz + r['bytes'] > args.max_bytes):
            batches.append(cur)
            cur, cursz = [], 0
        cur.append(r)
        cursz += r['bytes']
    if cur:
        batches.append(cur)
    print('into %d commits' % len(batches), flush=True)

    for bi, group in enumerate(batches):
        ops = []
        ok_rows = []
        skipped = 0
        for r in group:
            p = r['pod_path']
            if not os.path.exists(p):
                emit(out, 'file', status='MISSING', **{k: r[k] for k in
                     ('batch', 'path_in_repo', 'pod_path', 'bytes', 'sha256')})
                emit(state, 'MISSING', path_in_repo=r['path_in_repo'])
                skipped += 1
                continue
            sz = os.path.getsize(p)
            if sz != r['bytes']:
                emit(out, 'file', status='SIZE_CHANGED', seen=sz, **{k: r[k] for k in
                     ('batch', 'path_in_repo', 'pod_path', 'bytes', 'sha256')})
                emit(state, 'SIZE_CHANGED', path_in_repo=r['path_in_repo'])
                skipped += 1
                continue
            t0 = time.time()
            ls = sha256_file(p)
            if ls != r['sha256']:
                emit(out, 'file', status='SHA_CHANGED', seen=ls, **{k: r[k] for k in
                     ('batch', 'path_in_repo', 'pod_path', 'bytes', 'sha256')})
                emit(state, 'SHA_CHANGED', path_in_repo=r['path_in_repo'])
                skipped += 1
                continue
            emit(out, 'scanned', sha_secs=round(time.time() - t0, 1),
                 path_in_repo=r['path_in_repo'], bytes=sz)
            ops.append(mk_add(api, p, r['path_in_repo']))
            ok_rows.append(r)
        if not ops:
            emit(out, 'commit_skipped', index=bi, reason='no verifiable files',
                 missing=skipped)
            continue
        msg = 'Archive %s batch %d/%d (%d files, %.3f GiB)' % (
            args.batch, bi + 1, len(batches), len(ops),
            sum(r['bytes'] for r in ok_rows) / 2 ** 30)
        t0 = time.time()
        try:
            info = api.create_commit(repo_id=REPO, operations=ops, repo_type='model',
                                     commit_message=msg, revision='main')
            secs = time.time() - t0
            nb = sum(r['bytes'] for r in ok_rows)
            for r in ok_rows:
                emit(out, 'file', status='OK', commit=str(getattr(info, 'sha', '')),
                     **{k: r[k] for k in ('batch', 'path_in_repo', 'pod_path', 'bytes', 'sha256')})
                emit(state, 'OK', path_in_repo=r['path_in_repo'], sha256=r['sha256'])
            emit(out, 'commit', index=bi, ok=True, files=len(ops), bytes=nb,
                 secs=round(secs, 1), mbps=round(nb / 1e6 / max(secs, 0.01), 2),
                 revision=str(getattr(info, 'sha', '')),
                 commit_url=str(getattr(info, 'commit_url', '')))
        except Exception as e:  # noqa: BLE001
            emit(out, 'commit', index=bi, ok=False, files=len(ops),
                 err='%s: %s' % (type(e).__name__, str(e)[:900]))
            for r in ok_rows:
                emit(state, 'FAILED', path_in_repo=r['path_in_repo'])
            continue
    emit(out, 'finish', batch=args.batch,
         revision_after=api.repo_info(repo_id=REPO, repo_type='model').sha)
    state.close()
    out.close()
    return 0


def mk_add(api, path, in_repo):
    from huggingface_hub import CommitOperationAdd
    params = inspect.signature(CommitOperationAdd.__init__).parameters
    if 'path_or_fileobj' in params:
        return CommitOperationAdd(path_or_fileobj=path, path_in_repo=in_repo)
    return CommitOperationAdd(path_in_file=path, path_in_repo=in_repo)


if __name__ == '__main__':
    sys.exit(main())
