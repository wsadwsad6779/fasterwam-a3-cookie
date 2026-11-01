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
    '        row = run_correct(seed, variant, args.max_steps, args.noise, out, args.retries, tuple(args.recover_variants))')
p.write_text(t, encoding="utf-8")
print("patched")
PYEOF
python3 -m py_compile replay_correct.py && echo COMPILE_OK
sha256sum replay_correct.py
grep -n 'recover_variants' replay_correct.py
echo '--- csm4 full ---'
grep -c . /workspace/csm4.out
tail -c 1200 /workspace/csm4.out
echo __END__
