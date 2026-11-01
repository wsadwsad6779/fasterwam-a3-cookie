cat >> /workspace/_r5_stage/single_lowforce_probe.py <<'PY'

def run_seed(seed, max_steps, noise, out_dir):
    cfg = dataclasses.replace(load_config(CONFIG), horizon=6000)
    ct = cfg.cookie_transfer
    task = CookieTransferTaskConfig(
        cookie_count=len(ct.cookie_source_positions_m),
        position_noise_m=noise,
        yaw_noise_rad=0.015 if noise else 0.0,
        require_exact_slots=False,
        require_released=True,
        terminate_on_success=False,
    )
    env = A3CookieTransferEnv(
        config=cfg, task_config=task, action_mode="cartesian_delta", render_cameras=False
    )
    obs, info = env.reset(seed=int(seed), options={"randomize_cookies": bool(noise)})
    expert = SingleLowForce(env)
    expert.reset()

    xyz, eef, opening, applied, f18, wrist, in_target, names = [], [], [], [], [], [], [], []
    transitions = []
    t0 = time.time()
    stop = None
    for step in range(1, max_steps + 1):
        action = expert.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        f = np.asarray(obs[FORCE_KEY], dtype=np.float64)
        xyz.append(np.asarray(env.cookie_positions, dtype=np.float64).copy())
        eef.append(np.asarray(obs["observation.eef_pose"][:3], dtype=np.float64).copy())
        opening.append(float(obs["observation.state"][7]))
        applied.append(float(np.asarray(info["applied_action"])[7]))
        f18.append(f.copy())
        wrist.append(f[0:3].copy())
        in_target.append(int(info.get("cookies_in_target", -1)))
        if not names or expert.phase.name != names[-1][1]:
            names.append((step, expert.phase.name))
        if terminated:
            stop = "env_safety_stop: " + str(info.get("safety_reason"))
        elif truncated:
            stop = "horizon"
        elif expert.failed:
            stop = "expert_failed: " + str(expert.failure_reason)
        elif expert.finished:
            stop = "expert_done"
        if stop:
            break

    forces = np.asarray(f18)
    np.savez_compressed(
        out_dir / ("expert_single_%d.npz" % int(seed)),
        cookie_xyz=np.asarray(xyz),
        eef_xyz=np.asarray(eef),
        opening=np.asarray(opening),
        applied_opening=np.asarray(applied),
        pad_force_inner=forces[:, L_INNER],
        pad_force_outer=forces[:, L_OUTER],
        wrist_force=np.asarray(wrist),
        force18=forces,
        changed=np.zeros(len(xyz), dtype=bool),
        in_target=np.asarray(in_target),
        cookie_xyz0=np.asarray(info.get("cookie_positions", xyz[0]), dtype=np.float64),
    )
    summary = {
        "seed": int(seed),
        "noise_m": noise,
        "steps": len(xyz),
        "stop": stop,
        "success": bool(info.get("success", False)),
        "in_target_final": int(info.get("cookies_in_target", -1)),
        "exact_2x5_fill": bool(info.get("exact_2x5_fill", False)),
        "expert_max_pad_force": float(getattr(expert, "max_pad_force", 0.0)),
        "pad_peak_this_run": float(np.max(np.maximum(forces[:, L_INNER], forces[:, L_OUTER]))),
        "batch_reports": json.loads(json.dumps(expert.batch_reports, default=float)),
        "phase_transitions": names,
        "settings": {
            "batch_size": int(SingleLowForce.batch_size),
            "num_batches": int(SingleLowForce.num_batches),
            "target_grip_force": float(expert.target_grip_force),
            "stable_grip_force": float(expert.stable_grip_force),
            "min_grip_opening": float(expert.min_grip_opening),
            "secure_grip_opening": float(expert.secure_grip_opening),
            "release_opening": float(expert.release_opening),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    env.close()
    return summary
PY
wc -l /workspace/_r5_stage/single_lowforce_probe.py
echo DONE