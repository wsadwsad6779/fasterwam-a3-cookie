cd /workspace/_r5_stage
python3 - <<'PYEOF'
from pathlib import Path
p = Path("replay_correct.py")
t = p.read_text(encoding="utf-8")

def sub(old, new):
    global t
    n = t.count(old)
    assert n == 1, (n, old[:60])
    t = t.replace(old, new, 1)

sub('def run_correct(seed, variant, max_steps, noise, out_dir, retries=2):',
    'def run_correct(seed, variant, max_steps, noise, out_dir, retries=2, recover=(0,)):')
sub('''            variant = len(retry_log) - 1
''', '''            variant = recover[min(len(retry_log) - 1, len(recover) - 1)]
''')
sub('''    ap.add_argument("--out", required=True)
''', '''    ap.add_argument("--out", required=True)
    ap.add_argument("--recover-variants", type=int, nargs="+", default=[0])
''')
sub('        row = run_correct(seed, variant, args.max_steps, args.noise, out, args.retries)',
    '        row = run_correct(seed, variant, args.max_steps, args.noise, out, args.retries,
                           tuple(args.recover_variants))')
p.write_text(t, encoding="utf-8")
print("patched")
PYEOF
python3 -m py_compile replay_correct.py && echo COMPILE_OK
sha256sum replay_correct.py
grep -n 'recover' replay_correct.py
echo '--- csm4 last episode ---'
python3 -c "import json;rows=json.load(open('/workspace/a3_eval/demos_correct_smoke3/summary.json'));print(len(rows));print([[r['seed'],r['variant'],r['steps'],r['stop'],r['anomaly_observed'],r['accept'],[q['anomaly_kind'] for q in r['retry_log']]] for r in rows])"
echo __END__
