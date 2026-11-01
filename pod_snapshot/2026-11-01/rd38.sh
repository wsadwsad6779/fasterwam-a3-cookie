cd /workspace
ls data/a3_front_close_left_100
ls data/a3_front_close_left_100/data | head
ls data/a3_front_close_left_100/data/chunk-000 | head -3
ls data/a3_front_close_left_100/data/chunk-000 | wc -l
python3 - <<PY
import glob,pyarrow.parquet as pq,numpy as np
f=sorted(glob.glob("/workspace/data/a3_front_close_left_100/data/*/*.parquet"))
print(len(f))
t=pq.read_table(f[0])
print(t.schema.names)
print({k:str(v.type) for k,v in zip(t.schema.names,t.schema.types)})
PY
