'''Flatten backup_filelist.json + backup_hashes.jsonl into one TSV.

The manifest is rebuilt locally, so the two JSON blobs (1.7 MB together) are
merged here into a single 6-column table and hashed, which lets the download be
checked byte-for-byte instead of trusted.
'''
import json
import time
from pathlib import Path

OUT = Path('/workspace/_r6_stage/out')
d = json.loads((OUT / 'backup_filelist.json').read_text())
h = {}
err = []
for line in (OUT / 'backup_hashes.jsonl').read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    r = json.loads(line)
    if 'sha256' in r and r.get('sha256'):
        h[r['path']] = (r['sha256'], r.get('seconds', ''))
    elif 'error' in r:
        err.append((r['path'], r['error']))

rows = []
for e in d['entries']:
    sha, sec = h.get(e['path'], ('', ''))
    rows.append('\t'.join([
        e['path'], str(e['bytes']), e['mtime_utc'], e['suffix'],
        sha, str(e['referenced_by']), '|'.join(e['groups']), str(sec)]))

body = '\n'.join(rows) + '\n'
(OUT / 'manifest_rows.tsv').write_text(body)
print('rows %d bytes %d' % (len(rows), len(body)))
print('hashed %d unhashed %d hash_errors %d' % (
    sum(1 for r in rows if r.split(chr(9))[5]),
    sum(1 for r in rows if not r.split(chr(9))[5]), len(err)))
for p, m in err[:10]:
    print('ERR %s %s' % (p, m))
print('meta generated_utc %s entries_total_bytes %d' % (
    time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    sum(e['bytes'] for e in d['entries'])))
