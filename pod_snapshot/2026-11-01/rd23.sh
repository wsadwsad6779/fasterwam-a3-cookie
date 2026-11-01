cd /workspace
FW=/workspace/FasterWAM/src/fasterwam/models/wan22/fastwam.py
echo '=== checkpoint proprio ==='
grep -nE 'proprio_encoder|strict=' $FW | sed -n 20,60p
echo '=== loss composition ==='
grep -nE 'lambda_action|loss_action|loss_video|loss =|total_loss' $FW /workspace/FasterWAM/src/fasterwam/trainer.py | head -25
echo '=== lerobot rows and force ==='
python3 - <<'PY'
import glob
import pandas as pd
files = sorted(glob.glob('/workspace/data/a3_front_close_left_100/data/**/*.parquet', recursive=True))
print('parquet files', len(files))
if files:
    df = pd.read_parquet(files[0])
    print(files[0], df.shape)
    print(list(df.columns))
    if 'observation.force' in df.columns:
        f = df['observation.force']
        print('force elem type', type(f.iloc[0]), 'len', len(f.iloc[0]))
        print('first row', list(f.iloc[0]))
        print('max per dim', [round(float(max(f.iloc[i][j] for i in range(len(f)))), 3) for j in range(6)])
tot = sum(len(pd.read_parquet(p, columns=['timestamp'])) for p in files[:100])
print('rows in first 100 files', tot)
PY
ls /workspace/data/a3_front_close_left_100/data | head
tail -c 300 /workspace/gate10.out; tail -c 300 /workspace/gate.out
