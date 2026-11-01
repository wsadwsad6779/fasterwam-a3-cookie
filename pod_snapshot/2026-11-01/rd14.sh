cd /workspace
echo '=== r4_paired.sh ==='
cat /workspace/_r4_stage/r4_paired.sh
echo '=== windows.log stats ==='
wc -l /workspace/a3_train_runs/a3_lora_r5/windows.log
head -3 /workspace/a3_train_runs/a3_lora_r5/windows.log
awk '{s+=$2; if($2>m)m=$2} END {print sum_placeholder}' /dev/null 2>/dev/null
python3 - <<'PY'
import collections
c = collections.Counter()
idx = collections.Counter()
for line in open('/workspace/a3_train_runs/a3_lora_r5/windows.log'):
    parts = line.split()
    if len(parts) >= 2:
        idx[parts[0]] += 1
        c[int(parts[-1])] += 1
print('lines per first field:', dict(idx))
vals = []
for line in open('/workspace/a3_train_runs/a3_lora_r5/windows.log'):
    p = line.split()
    if len(p) >= 2:
        vals.append(int(p[-1]))
print('n', len(vals), 'min', min(vals), 'max', max(vals), 'mean', round(sum(vals)/len(vals),1))
PY
echo '=== who writes windows.log ==='
grep -rnE 'windows.log|WINDOW_LOG' /workspace/_r4_stage/up/r4_patch.new.py /workspace/FasterWAM/src/fasterwam 2>/dev/null | head -10
echo '=== sm2 tail ==='
tail -c 1500 /workspace/sm2.out
