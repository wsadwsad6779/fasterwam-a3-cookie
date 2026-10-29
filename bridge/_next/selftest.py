"""Local self-test: everything that does not need mujoco, torch or a GPU.

    python3 selftest.py

Covers the safety layer's no-op contract, the oracle counter, and the analysis
pass over a synthetic run. Run this after any edit to the npz keys.
"""
import json
import tempfile
from pathlib import Path

import numpy as np

import analyze_runs
import safety_wrap
from oracle_probe import OracleProbe


def fake_obs(force_in=0.0, force_out=0.0, xyz=(0.1, 0.3, 0.85)):
    force = np.zeros(18)
    force[safety_wrap.L_INNER] = force_in
    force[safety_wrap.L_OUTER] = force_out
    pose = np.r_[xyz, [1.0, 0.0, 0.0, 0.0], [0.0] * 7]
    return {"observation.eef_pose": pose, "observation.state": np.r_[np.zeros(7), 0.5],
            safety_wrap.FORCE_KEY: force}


def check(name, cond):
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(1)


def test_safety():
    print("safety layer")
    lim = safety_wrap.SafetyLimits()
    rng = np.random.default_rng(0)
    a = np.r_[rng.uniform(-1, 1, 6), 0.3, np.zeros(7)]
    st = safety_wrap.new_state()
    out = safety_wrap.apply(a, fake_obs(xyz=(0.15, 0.30, 0.90)), st, lim)
    check("normal action passes through untouched", np.array_equal(a, out))
    check("nothing counted as clipped", st["clipped"] == 0)

    st = safety_wrap.new_state()
    out = safety_wrap.apply(a, fake_obs(force_in=45.0), st, lim)
    check("45 N triggers the instant stop", st["estop"] and np.all(out[:6] == 0) and out[6] == 1.0)

    st = safety_wrap.new_state()
    obs = fake_obs(xyz=(0.095, 0.100, 0.760))
    for _ in range(3):
        out = safety_wrap.apply(np.r_[0.0, 0.0, -0.5, 0, 0, 0, -0.2, np.zeros(7)], obs, st, lim)
    check("empty gripper below the rim is pushed up", out[2] > 0.0 and st["binfloor"] == 3)

    st = safety_wrap.new_state()
    for _ in range(3):
        out = safety_wrap.apply(np.r_[0, 0, -0.5, 0, 0, 0, 0.0, np.zeros(7)],
                                fake_obs(force_in=30.0, xyz=(0.15, 0.30, 0.90)), st, lim)
    check("sustained 30 N retreats", out[2] >= lim.soft_retreat and st["soft"] >= 3)


def test_probe():
    print("oracle probe")

    class FakeEnv:
        cookie_positions = np.zeros((2, 3))

        def step(self):
            return "stepped"

    env = FakeEnv()
    probe = OracleProbe(env)
    check("reads pass through", probe.cookie_positions.shape == (2, 3))
    _ = probe.cookie_positions
    check("reads are counted", probe.counts["cookie_positions"] == 2)
    check("other attributes are not counted", probe.step() == "stepped" and probe.counts["data"] == 0)


def test_analysis():
    print("analysis")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        steps = 40
        pre = np.zeros((steps, 14))
        pre[:, 0] = 0.5
        post = pre.copy()
        post[10:20, 0] = 0.0          # wrapper overrides ten steps
        phase = np.full(steps, 0, dtype=np.int8)
        phase[10:20] = 2
        np.savez_compressed(d / "lora_full_11.npz", pre_14d=pre, post_14d=post,
                            chunk_idx=np.arange(steps), phase=phase,
                            opening=np.full(steps, 0.5), applied_opening=np.full(steps, 0.5),
                            pad_force_inner=np.zeros(steps), pad_force_outer=np.zeros(steps),
                            dist_to_source=np.zeros(steps), eef_xyz=np.zeros((steps, 3)),
                            oracle_cookie=np.arange(steps), oracle_data=np.zeros(steps),
                            oracle_sensor=np.zeros(steps))
        (d / "summary_lora.json").write_text(json.dumps([
            {"model": "lora", "control": "full", "seed": 11, "success": True,
             "cookies_in_target": 4}]))
        runs = analyze_runs.load(d)
        rec = runs[("lora", "full", 11)]
        check("verbatim counted per step", abs(rec["verbatim_pct"] - 75.0) < 1e-6)
        check("rewrite magnitude in mm", abs(rec["rewrite_mm_mean"] - (0.5 * 25.0 * 10 / 40)) < 1e-3)
        check("phase split found", rec["by_phase"]["descend"]["steps"] == 10 and
              rec["by_phase"]["open"]["verbatim_pct"] == 100.0)


def test_teacher_forcing_math():
    print("teacher forcing math")
    from scipy.spatial.transform import Rotation

    import run_teacher_forcing as tf

    rng = np.random.default_rng(0)
    steps = 40
    eef = np.zeros((steps, 7))
    eef[:, :3] = np.cumsum(rng.normal(0, 0.002, (steps, 3)), axis=0)
    eef[0, 3:7] = [1.0, 0.0, 0.0, 0.0]
    for t in range(1, steps):
        prev = Rotation.from_quat(np.r_[eef[t - 1, 4:7], eef[t - 1, 3]])
        q = (Rotation.from_rotvec(rng.normal(0, 0.004, 3)) * prev).as_quat()
        eef[t, 3] = q[3]
        eef[t, 4:7] = q[:3]

    opening = np.r_[np.ones(10), np.linspace(1.0, 0.4, 10), np.zeros(10),
                    np.linspace(0.4, 1.0, 10)]
    targets = tf.build_targets(eef, opening)
    check("one target per transition", targets.shape == (steps - 1, 7))

    # The label must invert the training formula exactly.
    R = tf._rotmats(eef)
    pos = eef[:1, :3] + np.vstack([np.zeros((1, 3)),
                                   np.cumsum(targets[:, :3] * tf.LIBERO_POS_M, axis=0)])
    rot_err = max(float(np.abs(Rotation.from_rotvec(targets[t, 3:6] * tf.LIBERO_ROT_RAD)
                            .as_matrix() @ R[t] - R[t + 1]).max()) for t in range(steps - 1))
    check("position / rotation labels invert the trajectory",
          np.abs(pos - eef[:, :3]).max() < 1e-12 and rot_err < 1e-9)

    g = targets[:, 6]
    check("gripper label is +1 close, -1 open, hold inside the deadband",
          g[2] == -1.0 and g[12] == 1.0 and g[22] == 1.0 and g[35] == -1.0)

    h = max(tf.HORIZONS) + 1
    act14 = np.zeros((steps, 14))
    act14[:, 6] = np.clip(2 * opening - 1, -1, 1)
    _, _, _, _, _, valid = tf.align(targets, np.ones((3, h), bool), opening,
                                   np.array([0, steps - 6, steps - 3]), act14)
    check("horizons past the trajectory end are dropped",
          valid.sum(axis=1).tolist() == [h, 5, 2])

    # An expert that commands exactly what the trajectory does must audit clean.
    act14 = np.zeros((steps, 14))
    act14[:, :3] = np.vstack([targets[:, :3] * (tf.LIBERO_POS_M / 0.025),
                              np.zeros((1, 3))])
    world = np.zeros((steps, 3))
    world[:-1] = targets[:, 3:6] * tf.LIBERO_ROT_RAD
    act14[:, 3:6] = np.einsum("tji,tj->ti", R, world) / 0.10
    act14[:, 6] = np.clip(2 * np.r_[opening[1:], opening[-1]] - 1, -1, 1)
    audit = tf.chain_audit(act14, eef, opening, (0.025, 0.10))
    check("chain audit inverts the adapter exactly",
          audit["pos_err_mm_p50"] < 1e-9 and audit["rot_err_mrad_p50"] < 1e-9
          and audit["grip_dir_agree_pct"] == 100.0)


if __name__ == "__main__":
    test_safety()
    test_probe()
    test_analysis()
    test_teacher_forcing_math()
    print("all local checks passed")
