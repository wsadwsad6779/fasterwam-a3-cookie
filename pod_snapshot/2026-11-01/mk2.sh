cd /workspace/_r5_stage
sed -i '/carry_peak_wrist_n/d; /env_close = None/d' replay_forces.py
sed -i 's|    expert.reset()|    expert.reset()
    z0_live = np.asarray(env.cookie_positions, dtype=np.float64).copy()|' replay_forces.py
sed -i 's|    z0 = np.asarray(info.get("cookie_positions", xyz[0]), dtype=np.float64)|    z0 = z0_live|' replay_forces.py
export PYTHONPATH=/workspace/a3_dual_arm_sim/src
python3 -m py_compile replay_forces.py && echo COMPILE_OK
grep -nE 'z0_live|z0 = |carry_peak_wrist|env_close' replay_forces.py
sha256sum replay_forces.py
