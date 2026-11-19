"""Same-state audit: teacher replay chain vs deployment chain on identical MuJoCo states.

One env, one expert trajectory (seed 41). At sampled frames we compute, from the SAME obs:
  * raw 14-wide eef_pose, state[7] opening, raw force18
  * the OLD collector proprio recipe (what teacher_replay_20261118 actually stored)
  * the deployment proprio14 (E.build_proprio14 = worker base8 + force-tare residual6)
  * processor-normalized versions of both
  * teacher image stack vs the deployment worker's prepared image tensor
  * the 7D action label from pose deltas and the Adapter's env-action mapping
  * the 16-frame P2 force history + mask
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

SIM_SRC = '/workspace/eval_restore_20261117/deploy/a3_dual_arm_sim/src'
if SIM_SRC not in sys.path:
    sys.path.insert(0, SIM_SRC)

import diag_reconstructed_40k as D

ROOT = Path(__file__).resolve().parent
E, factory, W, fw = D.setup_protocol()
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from force_stream import FORCE_OBS_KEY, select_channels, ForceStream
from action_adapter_v3 import FasterWAMA3ActionAdapterV3
from a3_dual_arm_sim.controllers.batch_expert import A3CookieBatchExpert

OUT = ROOT / 'c_class_eval_20261119' / 'state_contract_audit.json'
CALIB = E.read_json(str(D.CALIB))
PROC = None


def quat_wxyz_to_mat(q):
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rotvec_between(Ra, Rb):
    R = Rb @ Ra.T
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    ang = np.arccos(c)
    if ang < 1e-8:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return axis / (2 * np.sin(ang)) * ang


def old_propio(obs):
    eef = np.asarray(obs['observation.eef_pose'], dtype=np.float64)
    op = float(np.asarray(obs['observation.state'])[7])
    p = np.zeros(14, np.float32)
    p[0:3] = eef[:3]
    p[3:6] = rotvec_between(np.eye(3), quat_wxyz_to_mat(eef[3:7]))
    p[6] = 0.04 * op
    p[7] = -0.04 * op
    return p


def resize224(img):
    from PIL import Image
    a = np.asarray(img)
    if a.shape[0] == 224 and a.shape[1] == 224:
        return a.astype(np.uint8)
    return np.asarray(Image.fromarray(a).resize((224, 224), Image.BILINEAR), np.uint8)


def main():
    global PROC
    doc = OmegaConf.load(str(D.RUN_CFG))
    OmegaConf.set_struct(doc, False)
    if 'force' in doc.model and 'norm_stats' in doc.model.force:
        doc.model.force.norm_stats.file = str(D.CALIB)
    doc.data.train.pretrained_norm_stats = str(D.RECON)
    OmegaConf.set_struct(doc, True)
    stats = load_dataset_stats_from_json(str(D.RECON))
    PROC = instantiate(doc.data.train.processor).eval()
    PROC.set_normalizer_from_stats(stats)
    adapter = FasterWAMA3ActionAdapterV3()

    env = factory.make_env(position_noise_m=0.0, yaw_noise_rad=0.0, render_cameras=True)
    reset = D.make_reset(E, factory, env, SimpleNamespace(
        cpu=True, gripper_mode='pm1', max_steps=300, replan=10, allow_invalid_scene=False))
    obs = reset(env, 41)
    u = getattr(env, 'unwrapped', env)
    ex = A3CookieBatchExpert(env=u, action_mode='joint_position')
    ex.reset(context=None)
    stream = ForceStream(hist_len=16)
    snap_frames = {0, 30, 120, 240}
    snaps = []
    poses = []
    prev_obs = obs
    for t in range(300):
        eef = np.asarray(obs['observation.eef_pose'], np.float64)
        poses.append(eef)
        stream.push(obs[FORCE_OBS_KEY])
        if t in snap_frames:
            p_old = old_propio(obs)
            p14 = E.build_proprio14(obs, W, calib=CALIB, where='audit')
            n_old = W.normalize_proprio(p_old, PROC).detach().numpy().reshape(-1)
            n_new = W.normalize_proprio(p14, PROC).detach().numpy().reshape(-1)
            base8 = W.build_proprio(obs)
            raw6 = select_channels(obs[FORCE_OBS_KEY], where='audit').astype(np.float64)
            zero_ref = np.asarray(CALIB['zero_load_ref_median'], np.float64)
            teacher_img = np.concatenate([resize224(obs['observation.images.front']),
                                          resize224(obs['observation.images.left_wrist'])], axis=1)
            _rgb, dep_img = W.prepare_fasterwam_images_official(obs)
            dep = np.asarray(dep_img, dtype=np.float64)
            dep_raw_shape = list(dep.shape)
            if dep.ndim == 4:
                dep = dep[0]
            if dep.shape[0] == 3:
                dep = np.transpose(dep, (1, 2, 0))
            dep_scale = 'unit_0_1'
            if dep.min() < -0.01:
                dep = (dep + 1.0) * 127.5
                dep_scale = 'unit_-1_1'
            elif dep.max() <= 1.5:
                dep = dep * 255.0
            img_diff = np.abs(teacher_img.astype(np.float64) - dep)
            hist, mask = stream.snapshot()
            rec = {
                'frame': t,
                'raw_eef_pose14': eef.tolist(),
                'opening_state7': float(np.asarray(obs['observation.state'])[7]),
                'raw6_channels': raw6.tolist(),
                'zero_load_ref_median': zero_ref.tolist(),
                'deploy_base8': np.asarray(base8, np.float64).tolist(),
                'deploy_proprio14': p14.astype(np.float64).tolist(),
                'old_collector_proprio': p_old.astype(np.float64).tolist(),
                'col06_rel_diff': float(np.abs(p_old[:6] - p14[:6]).max()),
                'col67_old': p_old[6:8].tolist(), 'col67_deploy': p14[6:8].tolist(),
                'col67_max_abs_diff': float(np.abs(p_old[6:8] - p14[6:8]).max()),
                'col8_13_old': p_old[8:14].tolist(), 'col8_13_deploy': p14[8:14].tolist(),
                'col8_13_deploy_equals_raw_minus_zero': bool(
                    np.allclose(p14[8:14], raw6 - zero_ref, atol=1e-5)),
                'normalized_old': n_old.tolist(), 'normalized_new': n_new.tolist(),
                'normalized_old_absmax': float(np.abs(n_old).max()),
                'normalized_new_absmax': float(np.abs(n_new).max()),
                'image_teacher_shape': list(teacher_img.shape),
                'image_deploy_raw_shape': dep_raw_shape,
                'image_deploy_scale': dep_scale,
                'image_max_abs_diff': float(img_diff.max()),
                'image_mean_abs_diff': float(img_diff.mean()),
                'p2_history_tail': hist[-3:].tolist(), 'p2_mask_true': int(mask.sum()),
            }
            snaps.append(rec)
        a = ex.act(obs, '')
        out = env.step(np.asarray(a, np.float64))
        obs = out[0]
    if hasattr(env, 'close'):
        env.close()

    # action-label / Adapter algebra on two consecutive poses
    R0 = quat_wxyz_to_mat(poses[10][3:7])
    R1 = quat_wxyz_to_mat(poses[11][3:7])
    dp = poses[11][:3] - poses[10][:3]
    a7 = np.zeros(7)
    a7[:3] = dp / 0.05
    a7[3:6] = rotvec_between(R0, R1) / 0.5
    a7[6] = 1.0
    env_action = adapter.convert(a7, R0)
    action_check = {
        'pose_delta_m': dp.tolist(),
        'label_xyz_action_units': a7[:3].tolist(),
        'adapter_env_action_xyz': env_action[:3].tolist(),
        'env_action_x_0p025_m': (np.asarray(env_action[:3]) * 0.025).tolist(),
        'roundtrip_max_abs_m': float(np.abs(np.asarray(env_action[:3]) * 0.025 - dp).max()),
        'rot_label_rad_per_unit_0p5': a7[3:6].tolist(),
        'adapter_rot_out': env_action[3:6].tolist(),
        'grip_env_sign': float(env_action[6]),
    }

    out = {
        **D.MARKERS, 'TRAINING_DATA_IDENTITY': 'CANDIDATE_RECONSTRUCTED_TRAINING_DATA',
        'TEACHER_DATA_IDENTITY': 'CANDIDATE_TEACHER_DATA',
        'seed': 41, 'snapshots': snaps, 'action_adapter_algebra': action_check,
        'findings': {
            'raw_eef_identical_between_chains': True,
            'old_proprio_cols_0_6_source': 'eef xyz + rotvec from the same obs (columns 0..5)',
            'old_proprio_cols_6_7': '+-0.04*opening, NOT the two finger joint values',
            'old_proprio_cols_8_13': 'hard zeros: the collector never added the force-tare residual',
            'why_17_sigma': 'arrays.npz ep_start was [0,1,2,3] (episode count, not cumulative '
                            'frames) and proprio was 75% zero rows; means were computed over that '
                            'corrupt array, not over a different coordinate frame',
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    brief = [{k: s[k] for k in ('frame', 'col06_rel_diff', 'col67_max_abs_diff',
                                'col8_13_deploy_equals_raw_minus_zero',
                                'normalized_old_absmax', 'normalized_new_absmax',
                                'image_teacher_shape', 'image_deploy_raw_shape',
                                'image_max_abs_diff', 'image_mean_abs_diff', 'p2_mask_true')} for s in snaps]
    print(json.dumps({'snapshots': brief, 'action': action_check}, indent=2))
    print('AUDIT_DONE', flush=True)


if __name__ == '__main__':
    main()
