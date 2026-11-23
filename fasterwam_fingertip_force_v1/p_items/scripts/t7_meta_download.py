import json
import os
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

D = '/workspace/fwv1/data/hf_meta'
OUT = '/workspace/fwv1/p_items/DATASET_800_META_STATE.json'
os.makedirs(D, exist_ok=True)
ok = False
for attempt in range(1, 7):
    try:
        snapshot_download(repo_id='Eter0109/a3-cookie-multi-tasks', repo_type='dataset',
                          local_dir=D, allow_patterns=['meta/**'], max_workers=2)
        ok = True
        break
    except Exception as ex:
        print('RETRY', attempt, type(ex).__name__, str(ex)[:150], flush=True)
        time.sleep(20 * attempt)
files = []
for root, _, names in os.walk(D):
    for n in names:
        fp = os.path.join(root, n)
        files.append({'path': os.path.relpath(fp, D), 'bytes': os.path.getsize(fp)})
json.dump({'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'ok': ok, 'files': files},
          open(OUT, 'w'), indent=1)
print(json.dumps({'ok': ok, 'n': len(files), 'paths': [f['path'] for f in files]}))