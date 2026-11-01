import hashlib, shutil
import py_compile

F = '/workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py'
BAK = '/workspace/_r4_stage/run_matrix_pre_b0.py'
CR, LF = chr(13), chr(10)
src = open(F, 'rb').read().decode('utf-8').replace(CR + LF, LF)
shutil.copyfile(F, BAK)


def blk(s):
    return s[1:]


EDITS = [
 (blk(r'''
    control in {C0, C1..C5, safety, force, guard, full}
        C0 / safety : interface + safety layer only
        C1..C5      : full wrapper, one capability handed back per stage
                      (transport xyz -> approach xyz -> grasp target ->
                       descend/close -> release/recovery); C5 is all of them and
                      must reproduce C0
'''),
  blk(r'''
    control in {B0, C0, C1..C5, safety, force, guard, full}
        B0            : bare policy -- interface only, safety layer off. The
                        default condition; always paired with C0.
        C0 / safety   : interface + safety layer only
        C1..C5        : full wrapper, one capability handed back per stage
                        (transport xyz -> approach xyz -> grasp target ->
                         descend/close -> release/recovery); C5 is all of them and
                        must reproduce C0
''')),
 (blk(r'''
Authority moves one capability at a time, never by adding task rules.
'''),
  blk(r'''
Authority moves one capability at a time, never by adding task rules.

B0 and C0 are the same episode with and without the safety layer: same weights,
seed, reset and action stream, so the only difference is what the layer rewrote.
That difference is not small -- r4's C0 seed 22 scored 3/10 with 112 of 600
steps rewritten -- so every run reports both, and the B0 record carries
`safety_would_have_*` counters for the intervention the layer would have made.
''')),
 (blk(r'''
    python3 run_matrix.py --model lora --controls C0,C1,C2 --seeds 11 21 22
'''),
  blk(r'''
    python3 run_matrix.py --model lora --controls B0,C0 --seeds 11 21 22
''')),
 (blk(r'''
    """Map a control name to (recorded name, wrapped?, profile, released caps).

    `safety` and `C0` are the same condition: interface + safety layer only.
'''),
  blk(r'''
    """Map a control name to (recorded name, wrapped?, profile, released caps).

    `B0` is the bare policy: interface only (frame adapter + pm1 gripper
    binarisation), no safety layer. It is the default condition from here on, and
    it is paired with `C0` on the same seed so the intervention rate stops being
    invisible.

    `safety` and `C0` are the same condition: interface + safety layer only.
''')),
 (blk(r'''
    upper = name.upper()
    if name.lower() == "safety" or upper == "C0":
'''),
  blk(r'''
    upper = name.upper()
    if upper == "B0":
        return "B0", False, a3_wrap.PROFILES["full"], frozenset()
    if name.lower() == "safety" or upper == "C0":
''')),
 (blk(r'''
    safety_st = safety_wrap.new_state() if staged else None
'''),
  blk(r'''
    safety_st = safety_wrap.new_state() if staged else None
    # B0 executes nothing but the interface output, but the safety layer is
    # still run on a throwaway copy so the intervention it *would* have made is
    # counted instead of silently absent.
    bare = control_name == "B0"
    shadow_st = safety_wrap.new_state() if bare else None
    shadow_changed = []
    nan_abort = -1
''')),
 (blk(r'''
            post_w = pre.copy()
            post = safety_wrap.apply(pre, obs, ctl_state, lim)
            phase_log.append(-1)
'''),
  blk(r'''
            post_w = pre.copy()
            if bare:
                post = pre.copy()
                shadow_changed.append(bool(np.any(
                    safety_wrap.apply(pre.copy(), obs, shadow_st, lim) != pre)))
            else:
                post = safety_wrap.apply(pre, obs, ctl_state, lim)
            phase_log.append(-1)
''')),
 (blk(r'''
        obs, _, terminated, truncated, info = env.step(post)
        tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
'''),
  blk(r'''
        obs, _, terminated, truncated, info = env.step(post)
        tcp = np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64)
        if not np.all(np.isfinite(tcp)):
            # Nothing is being clamped, so an unstable state has to end the
            # episode and be reported instead of quietly poisoning the log.
            nan_abort = step
            print(f"[{policy.name}/{control}/{seed}] non-finite eef at "
                  f"step={step + 1}, aborting episode", flush=True)
            break
''')),
 (blk(r'''
        "safety_layer": bool(staged or not wrapped),
'''),
  blk(r'''
        "safety_layer": bool((staged or not wrapped) and not bare),
''')),
 (blk(r'''
        "binfloor_steps": int(ctl_state.get("binfloor", 0)),
'''),
  blk(r'''
        "binfloor_steps": int(ctl_state.get("binfloor", 0)),
        # B0 only: what the safety layer would have rewritten had it been in the
        # loop. Counted on a throwaway copy, never executed.
        "safety_would_have_changed_pct": round(100.0 * float(np.mean(shadow_changed)), 2)
                                        if bare and shadow_changed else None,
        "safety_would_have_hard": int(shadow_st.get("hard", 0)) if bare else None,
        "safety_would_have_binfloor": int(shadow_st.get("binfloor", 0)) if bare else None,
        "nan_abort_step": int(nan_abort),
''')),
 (blk(r'''
    ap.add_argument("--controls", default="C0,C1,C2",
                    help="C0/safety = interface + safety only; C1..C5 = one "
                         "capability released per stage (C5 releases all and "
                         "must match C0); full/guard/force = historical profiles")
'''),
  blk(r'''
    ap.add_argument("--controls", default="B0,C0",
                    help="B0 = bare policy, interface only, no safety layer "
                         "(the default condition, paired with C0 on the same "
                         "seeds); C0/safety = interface + safety only; C1..C5 = "
                         "one capability released per stage (C5 releases all and "
                         "must match C0); full/guard/force = historical profiles")
''')),
]

fail = []
for i, (old, new) in enumerate(EDITS):
    n = src.count(old)
    if n != 1:
        fail.append((i, n, old.replace(LF, ' | ')[:90]))
        continue
    src = src.replace(old, new)
if fail:
    for f in fail:
        print('ANCHOR_FAIL', f)
    raise SystemExit(1)

out = src.replace(LF, CR + LF).encode('utf-8')
open(F, 'wb').write(out)
print('SHA256', hashlib.sha256(out).hexdigest())
py_compile.compile(F, doraise=True)
print('COMPILE_OK')
