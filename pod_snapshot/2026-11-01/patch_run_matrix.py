import pathlib
D = pathlib.Path('/workspace/fasterwam_a3_bridge/bridge/_next')
P = D / 'run_matrix.py'
src = P.read_text().replace('\r\n', '\n')
(D / 'run_matrix_orig_check.py').write_text(src)
R = []
R.append(("import a3_wrap\n", "import a3_wrap\nimport cookie_truth_log\n"))
R.append(("""        if wrapped:
            post = a3_wrap.apply(pre.copy(), obs, probe, ctl_state, profile, released)
            if staged:
                post = safety_wrap.apply(post, obs, safety_st, lim)""",
"""        # post_w keeps the wrapper output before the safety layer, so the two
        # intervention rates are attributable. All three arrays are 14-d in the
        # same units and at the same timestep.
        if wrapped:
            post_w = a3_wrap.apply(pre.copy(), obs, probe, ctl_state, profile, released)
            post = safety_wrap.apply(post_w, obs, safety_st, lim) if staged else post_w"""))
R.append(("""        else:
            post = safety_wrap.apply(pre, obs, ctl_state, lim)""",
"""        else:
            post_w = pre.copy()
            post = safety_wrap.apply(pre, obs, ctl_state, lim)"""))
R.append(("    applied_canonical = []", "    applied_canonical, post_wrap_log, cookie_log = [], []"))
R.append(("    policy.calls = 0", "    policy.calls = 0\n    cookie_xyz0 = cookie_truth_log.initial(env)"))
R.append(("""        pre_log.append(pre)
        post_log.append(np.asarray(post, dtype=np.float64))""",
"""        pre_log.append(pre)
        post_log.append(np.asarray(post, dtype=np.float64))
        post_wrap_log.append(np.asarray(post_w, dtype=np.float64))"""))
R.append(("        eef.append(tcp)",
"""        eef.append(tcp)
        # Diagnostics only: same timestep as tcp, never fed back to any controller.
        cookie_log.append(cookie_truth_log.snapshot(env, step))"""))
R.append(("        changed=changed,",
"""        changed=changed,
        post_wrap_14d=np.asarray(post_wrap_log),
        cookie_xyz=np.asarray([c for _, c in cookie_log]),
        cookie_step=np.asarray([s for s, _ in cookie_log]),
        cookie_xyz0=cookie_xyz0,"""))
R.append(('        "control": control_name,', '        "control": control_name,\n        "wrapped": bool(wrapped),'))
R.append(("    args = parse_args()",
"""    args = parse_args()
    if args.model == "lora" and not args.ckpt:
        raise SystemExit("refusing to run lora without an explicit --ckpt")"""))
R.append(("    lim = make_limits(args)",
"""    ck = manifest["checkpoint"]
    print("[matrix] loaded_ckpt=%s size=%s sha256_head=%s"
          % (ck["path"], ck["size"], ck["sha256_head"]), flush=True)

    lim = make_limits(args)"""))
for i, (o, n) in enumerate(R):
    c = src.count(o)
    if c != 1:
        raise SystemExit("REPLACE_%d_COUNT_%d" % (i, c))
    src = src.replace(o, n)
P.write_text(src)
print("PATCH_OK", len(R))
