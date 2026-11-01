cd /workspace
echo '=== r4 b0/c0 records ==='
python3 - <<'PY'
import json, glob
for p in sorted(glob.glob('/workspace/a3_eval/r4_b0_c0_paired/summary_*.json')) + sorted(glob.glob('/workspace/a3_eval/r4_c0_frozen/summary_*.json')):
    rs = json.load(open(p))
    print(p, len(rs))
    for r in rs:
        print('  ', r.get('model'), r.get('control'), r.get('seed'), 'succ', r.get('success'), 'in_target', r.get('cookies_in_target'), 'nan_abort', r.get('nan_abort_step'), 'verbatim', r.get('verbatim_pct'), 'would_chg', r.get('safety_would_have_changed_pct'), 'would_hard', r.get('safety_would_have_hard'), 'maxpad', r.get('max_pad_force'))
PY
echo '=== estop keys ==='
grep -rlE 'safety_reason' /workspace/a3_eval/r4_b0_c0_paired 2>/dev/null | head
ls /workspace/a3_eval | head -30
df -h /workspace | tail -1
