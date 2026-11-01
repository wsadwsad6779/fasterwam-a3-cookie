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

sub('''        if expert.finished:
            stop = "expert_done"
            break
        if expert.failed:
''', '''        if expert.failed:
''')

sub('''            expert.variant_index = variant
            expert.reset()
''', '''            expert.variant_index = variant
            expert.reset()
        if expert.finished:
            stop = "expert_done"
            break
''')

sub('''            over = int(sum(1 for v in window if v > ANOMALY_PAD_N))
''', '''            over = int(sum(1 for v in window if v > ANOMALY_PAD_N))
            kind = ("excessive_pad_force" if peak > ANOMALY_PAD_N and over >= ANOMALY_FRAMES
                    else ("empty_grasp" if peak < 0.3 else "stuck_no_force"))
''')

sub('''                "frames_over_2n": int(sum(1 for v in window if v > 2.0)),
''', '''                "frames_over_2n": int(sum(1 for v in window if v > 2.0)),
                "anomaly_kind": kind,
''')

sub('''        "anomaly_observed": bool(anomaly),
''', '''        "anomaly_observed": bool(anomaly),
        "force_anomaly": any(r["anomaly_kind"] == "excessive_pad_force" for r in retry_log),
''')

p.write_text(t, encoding="utf-8")
print("patched", len(t))
PYEOF
python3 -m py_compile replay_correct.py && echo COMPILE_OK
sha256sum replay_correct.py
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
cd /workspace/fasterwam_a3_bridge/bridge/_next
date
python3 -u /workspace/_r5_stage/replay_correct.py --seeds 7204 7215 7203 --variants 3 14 2 --noise 0.0003 --retries 1 --max-steps 1600 --out /workspace/a3_eval/demos_correct_smoke3
date
echo __END__
