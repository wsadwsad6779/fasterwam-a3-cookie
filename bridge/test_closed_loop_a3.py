"""Closed-loop test: FasterWAM adapters + A3 cookie transfer environment.

Uses real FasterWAM model output (from NPZ) to drive the A3 simulation,
validating the full observation -> action pipeline without a GPU.

Tests:
1. Vision adapter: A3 cameras -> model input format
2. Observation adapter: A3 state -> model proprio format
3. Action adapter: model 7D -> A3 14D (with gripper binarization fix)
4. Full closed-loop: obs -> adapters -> action -> env.step()
"""

import sys
import time
import numpy as np

sys.path.insert(0, ".")
sys.path.insert(0, "../a3_dual_arm_sim/src")

from a3_dual_arm_sim.tasks.cookie_transfer import A3CookieTransferEnv
from action_adapter_v3 import FasterWAMA3ActionAdapterV3
from vision_adapter_official import prepare_fasterwam_images_official
from observation_adapter import extract_left_eef_state


def load_real_model_output(npz_path):
    """Load real FasterWAM action chunk from NPZ."""
    data = np.load(npz_path, allow_pickle=True)
    actions = data["actions"]
    print(f"Loaded model output: {actions.shape}, range=[{actions.min():.4f}, {actions.max():.4f}]")
    return actions


def test_vision_adapter(env):
    """Test: A3 camera images -> FasterWAM model input."""
    obs, _ = env.reset(seed=0)
    rgb, x = prepare_fasterwam_images_official(obs)
    assert rgb.shape == (224, 448, 3), f"RGB shape: {rgb.shape}"
    assert rgb.dtype == np.uint8
    assert x.shape == (1, 3, 224, 448), f"Model input shape: {x.shape}"
    assert x.dtype == np.float32
    assert x.min() >= -1.0 and x.max() <= 1.0, f"Range: [{x.min()}, {x.max()}]"
    print(f"PASS: vision adapter -> rgb={rgb.shape}, x={x.shape}, range=[{x.min():.2f}, {x.max():.2f}]")
    return obs


def test_observation_adapter(obs):
    """Test: A3 eef_pose -> model proprio (6D left EEF state)."""
    eef_state = extract_left_eef_state(obs)
    assert eef_state.shape == (6,), f"EEF state shape: {eef_state.shape}"
    assert eef_state.dtype == np.float32
    assert np.all(np.isfinite(eef_state))
    print(f"PASS: observation adapter -> eef_state={eef_state}")
    return eef_state


def test_action_adapter_with_real_output(action_chunk, R):
    """Test: real FasterWAM 7D actions -> A3 14D, verify no overflow."""
    adapter = FasterWAMA3ActionAdapterV3()
    converted = []
    gripper_values_before = []
    gripper_values_after = []

    for i, action7 in enumerate(action_chunk):
        g_before = action7[6]
        gripper_values_before.append(g_before)
        result = adapter.convert(action7, R)
        gripper_values_after.append(result[6])
        converted.append(result)

    converted = np.array(converted)
    gripper_before = np.array(gripper_values_before)
    gripper_after = np.array(gripper_values_after)

    overflow_mask = np.abs(gripper_before) > 1.0
    overflow_count = np.sum(overflow_mask)

    print(f"PASS: action adapter converted {len(action_chunk)} actions")
    print(f"  Input  gripper range: [{gripper_before.min():.4f}, {gripper_before.max():.4f}]")
    print(f"  Output gripper range: [{gripper_after.min():.4f}, {gripper_after.max():.4f}]")
    print(f"  Output 14D range: [{converted.min():.4f}, {converted.max():.4f}]")
    print(f"  Gripper values that would have overflowed: {overflow_count}/{len(action_chunk)}")
    if overflow_count > 0:
        print(f"  Before fix, these would crash: {gripper_before[overflow_mask]}")
        print(f"  After fix (binarized): {gripper_after[overflow_mask]}")

    assert np.all(np.abs(converted) <= 1.0 + 1e-6), f"Output exceeds limits!"
    return converted


def test_closed_loop(env, action_chunk, max_steps=50):
    """Full closed-loop: reset env, execute actions step by step."""
    adapter = FasterWAMA3ActionAdapterV3()

    obs, info = env.reset(seed=0)
    print(f"\nClosed-loop test: {max_steps} steps, {len(action_chunk)} model actions")
    print(f"  Initial cookies_in_target: {info.get('cookies_in_target', 'N/A')}")

    R = np.eye(3, dtype=np.float64)
    action_index = 0
    replan_count = 0

    for step in range(max_steps):
        if action_index >= len(action_chunk):
            action_index = 0
            replan_count += 1

        model_action = action_chunk[action_index]
        a3_action = adapter.convert(model_action, R)
        obs, reward, terminated, truncated, info = env.step(a3_action)

        if step % 10 == 0:
            print(f"  step={step:03d} gripper={a3_action[6]:+.1f} "
                  f"cookies_in_target={info.get('cookies_in_target', 'N/A')} "
                  f"terminated={terminated}")

        if terminated or truncated:
            print(f"  Episode ended at step {step}: terminated={terminated}, truncated={truncated}")
            break

        action_index += 1

    print(f"\nClosed-loop result:")
    print(f"  Steps executed: {step + 1}")
    print(f"  Replans: {replan_count}")
    print(f"  Cookies in target: {info.get('cookies_in_target', 'N/A')}")
    print(f"  Success: {info.get('success', False)}")
    print(f"  Safety stop: {info.get('safety_stop', False)}")

    return info


if __name__ == "__main__":
    npz_path = "../FasterWAM/a3_left_pickup_actions.npz"
    max_steps = 50

    print("=" * 60)
    print("Closed-loop test: FasterWAM adapters + A3 cookie env")
    print("=" * 60)

    action_chunk = load_real_model_output(npz_path)

    env = A3CookieTransferEnv(
        action_mode="cartesian_delta",
        render_mode=None,
        render_cameras=True,
    )

    print("\n--- Test 1: Vision adapter ---")
    obs = test_vision_adapter(env)

    print("\n--- Test 2: Observation adapter ---")
    eef_state = test_observation_adapter(obs)

    print("\n--- Test 3: Action adapter with real model output ---")
    R = np.eye(3, dtype=np.float64)
    converted = test_action_adapter_with_real_output(action_chunk, R)

    print("\n--- Test 4: Full closed-loop ---")
    info = test_closed_loop(env, action_chunk, max_steps=max_steps)

    env.close()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
