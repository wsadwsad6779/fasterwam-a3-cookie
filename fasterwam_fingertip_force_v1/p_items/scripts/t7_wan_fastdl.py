import os, sys, json, time, hashlib, subprocess, traceback

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
D = '/workspace/fwv1/wan_base'
P = '/workspace/fwv1/p_items'
LOG = P + '/job_wan_t7.log'
STATE = P + '/POD_JOB_STATE.json'
LIST = P + '/wan_filelist.json'
LEDGER = P + '/WAN_VERIFIED.json'
OUT = P + '/P4_WAN_FASTDL.json'
BASEURL = 'https://modelscope.cn/api/v1/models/Wan-AI/Wan2.2-TI2V-5B/repo?Revision=master&FilePath='
NONESSENTIAL = {'.gitattributes', 'README.md', 'configuration.json', '.msc', '.mv'}
START = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def log(m):
    print(m, flush=True)
    with open(LOG, 'a') as f:
        f.write(m + '\n')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def st(status, note='', extra=None):
    d = {'started_at': START, 'ts': utc(), 'pid': os.getpid(), 'stage': 'wan_download',
         'log_path': LOG, 'status': status, 'expected_artifacts': [D],
         'next': 'P4 full smoke (local Wan dir + official step_021700)', 'note': note}
    if extra:
        d.update(extra)
    with open(STATE, 'w') as f:
        json.dump(d, f, indent=1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 22), b''):
            h.update(c)
    return h.hexdigest()


def fetch_list():
    if os.path.exists(LIST):
        try:
            j = json.load(open(LIST))
            if j.get('files'):
                return j['files'], j.get('source', 'cache')
        except Exception:
            pass
    last = ''
    for attempt in range(1, 7):
        try:
            from huggingface_hub import HfApi
            tree = HfApi().list_repo_tree('Wan-AI/Wan2.2-TI2V-5B', repo_type='model', recursive=True)
            files = []
            for e in tree:
                p = getattr(e, 'path', None)
                s = getattr(e, 'size', None)
                l = getattr(e, 'lfs', None)
                if p and s is not None:
                    files.append({'path': p, 'size': int(s),
                                  'sha256': (getattr(l, 'sha256', None) if l is not None else None)})
            json.dump({'source': 'hf_api', 'ts': utc(), 'files': files}, open(LIST, 'w'), indent=1)
            return files, 'hf_api'
        except Exception as ex:
            last = repr(ex)[:200]
            log('LIST_RETRY %d %s' % (attempt, last))
            time.sleep(20)
    try:
        import requests
        u = 'https://modelscope.cn/api/v1/models/Wan-AI/Wan2.2-TI2V-5B/repo/files?Revision=master&Recursive=True'
        j = requests.get(u, timeout=30).json()
        data = j.get('Data') or {}
        rows = data.get('Files') or data.get('files') or []
        files = []
        for r in rows:
            p = r.get('Path') or r.get('Name')
            s = r.get('Size')
            h = r.get('Sha256') or r.get('SHA256')
            if p and s is not None and r.get('Type') in (None, 'blob'):
                files.append({'path': p, 'size': int(s), 'sha256': h})
        if files:
            json.dump({'source': 'modelscope_api', 'ts': utc(), 'files': files}, open(LIST, 'w'), indent=1)
            return files, 'modelscope_api'
    except Exception as ex:
        last += ' | ms:' + repr(ex)[:150]
    raise RuntimeError('file list unavailable: ' + last)


def main():
    os.makedirs(D, exist_ok=True)
    st('running', note='task7 fastdl: resume any missing file, verify size+sha256 with ledger')
    log('START ' + START)
    files, src = fetch_list()
    log('FILES=%d src=%s' % (len(files), src))
    ledger = {}
    if os.path.exists(LEDGER):
        try:
            ledger = json.load(open(LEDGER))
        except Exception:
            ledger = {}
    results = []
    for f in files:
        rel = f['path']
        want = int(f['size'])
        sha = f.get('sha256')
        dst = os.path.join(D, rel)
        d = os.path.dirname(dst)
        if d:
            os.makedirs(d, exist_ok=True)
        rec = {'path': rel, 'want_size': want, 'hf_sha256': sha}
        sz0 = os.path.getsize(dst) if os.path.exists(dst) else 0
        rec['bytes_before'] = sz0
        t0 = time.time()
        if sz0 != want:
            url = BASEURL + rel
            try:
                r = subprocess.run(['curl', '-sL', '--retry', '8', '--retry-all-errors',
                                    '-C', '-', '-o', dst, url], timeout=10800)
                rec['curl_rc'] = r.returncode
            except Exception as ex:
                rec['curl_exc'] = repr(ex)[:200]
        sz = os.path.getsize(dst) if os.path.exists(dst) else 0
        sec = round(time.time() - t0, 2)
        rec['size'] = sz
        rec['sec'] = sec
        if sec > 0:
            rec['MBps'] = round(max(0, sz - sz0) / 1e6 / max(sec, 1e-3), 2)
        good = (sz == want)
        if good and sha:
            led = ledger.get(rel) or {}
            if led.get('sha256') == sha and led.get('size') == want:
                rec['sha256'] = sha
                rec['sha_check'] = 'ledger_match'
            else:
                h = sha256_file(dst)
                rec['sha256'] = h
                rec['sha_check'] = 'match' if h == sha else 'MISMATCH'
                good = (h == sha)
                if good:
                    ledger[rel] = {'size': sz, 'sha256': h, 'ts': utc()}
                    json.dump(ledger, open(LEDGER, 'w'), indent=1)
        elif good and not sha:
            rec['sha_check'] = 'no_expected_sha'
        else:
            rec['sha_check'] = 'size_mismatch'
        rec['essential'] = not (rel in NONESSENTIAL or rel.startswith('assets/'))
        rec['ok'] = bool(good)
        results.append(rec)
        speed = (str(rec.get('MBps')) + 'MB/s') if 'MBps' in rec else ''
        log(('OK ' if good else 'BAD ') + rel + ' size=%d/%d %s %s' % (sz, want, rec['sha_check'], speed))
    bad = [r for r in results if not r['ok']]
    bad_ess = [r for r in bad if r['essential']]
    total = sum(r['size'] for r in results)
    summary = {
        'worker': 'task7', 'ts': utc(), 'source': 'modelscope', 'filelist_source': src,
        'method': 'curl -sL --retry 8 --retry-all-errors -C - per file; size + HF LFS sha256 (ledger-cached)',
        'dir': D, 'files_total': len(results), 'files_ok': len(results) - len(bad),
        'files_bad': len(bad), 'essential_bad': len(bad_ess),
        'bytes_total': total, 'bytes_want': sum(r['want_size'] for r in results),
        'nonessential_notes': [r['path'] for r in bad if not r['essential']],
        'results': results, 'log': LOG,
    }
    json.dump(summary, open(OUT, 'w'), indent=1)
    st('done' if not bad_ess else 'partial',
       note='ok=%d bad=%d essential_bad=%d bytes=%d' % (len(results) - len(bad), len(bad), len(bad_ess), total),
       extra={'files_ok': len(results) - len(bad), 'files_bad': len(bad), 'total_bytes': total})
    log('DONE ok=%d bad=%d essential_bad=%d total=%d' % (len(results) - len(bad), len(bad), len(bad_ess), total))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        st('failed', note=traceback.format_exc()[-600:])
        log('TOPFAIL')
        log(traceback.format_exc()[:2000])
        sys.exit(1)