import json
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import HfApi

OUT = '/workspace/fwv1/p_items/DATASET_800_VIDEO_TREE.json'
api = HfApi()
res = {'repo': 'Eter0109/a3-cookie-multi-tasks', 'repo_type': 'dataset'}
try:
    tree = api.list_repo_tree('Eter0109/a3-cookie-multi-tasks', repo_type='dataset', recursive=True)
    files = []
    for e in tree:
        p = getattr(e, 'path', None)
        s = getattr(e, 'size', None)
        if p and s is not None:
            files.append({'path': p, 'size': int(s)})
    res['files'] = len(files)
    res['total_bytes'] = sum(f['size'] for f in files)
    groups = {}
    for f in files:
        top = f['path'].split('/')[0]
        g = groups.setdefault(top, {'files': 0, 'bytes': 0})
        g['files'] += 1
        g['bytes'] += f['size']
    res['groups'] = groups
    res['video_sample'] = [f for f in files if f['path'].startswith('videos/')][:10]
except Exception as ex:
    res['error'] = type(ex).__name__ + ': ' + str(ex)[:300]
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps({k: res.get(k) for k in ['files', 'total_bytes', 'groups', 'error']}))