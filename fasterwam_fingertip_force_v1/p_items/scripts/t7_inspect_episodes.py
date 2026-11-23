import glob
import json
import os

import pyarrow.parquet as pq

P = '/workspace/fwv1/data/hf_meta/meta/episodes/chunk-000/file-000.parquet'
res = {'parquet': P}
t = pq.read_table(P)
res['rows'] = t.num_rows
res['columns'] = t.column_names
row = {k: str(t[k][0].as_py())[:160] for k in t.column_names}
res['row0'] = row
res['wan_dirs'] = sorted(os.listdir('/workspace/fwv1/wan_base'))
tok = []
for pat in ('**/spiece.model', '**/tokenizer.json', '**/tokenizer_config.json'):
    tok += glob.glob('/workspace/fwv1/wan_base/' + pat, recursive=True)
res['tokenizer_files'] = [os.path.relpath(x, '/workspace/fwv1/wan_base') for x in tok[:8]]
res['t5_bytes'] = os.path.getsize('/workspace/fwv1/wan_base/models_t5_umt5-xxl-enc-bf16.pth') if os.path.exists('/workspace/fwv1/wan_base/models_t5_umt5-xxl-enc-bf16.pth') else None
json.dump(res, open('/workspace/fwv1/p_items/DATASET_800_EPISODES_INSPECT.json', 'w'), indent=1)
print(json.dumps({k: res[k] for k in ['rows', 'columns', 'tokenizer_files', 't5_bytes']}))
print(json.dumps(row)[:1200])