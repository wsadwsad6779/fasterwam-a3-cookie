import os
H = {}
for line in open('/workspace/a3_eval/disk_hashes.txt'):
    if line.startswith('DONE'):
        continue
    parts = line.split(None, 1)
    if len(parts) != 2:
        continue
    H[parts[1].strip()] = parts[0]
S = {}
for line in open('/workspace/a3_eval/disk_files.txt'):
    size, _, path = line.rstrip('\n').partition('\t')
    S[path] = int(size)
KEYS = [
    '/workspace/a3_data/a3_fw/frames.dat',
    '/workspace/a3_data/a3_fw/dataset_stats.json',
    '/workspace/a3_data/a3_fw_dagger/frames.dat',
    '/workspace/a3_data/a3_fw_dagger_near/frames.dat',
    '/workspace/a3_train_runs/a3_lora_r1/step_002000_merged.pt',
    '/workspace/a3_train_runs/a3_lora_r2/step_003000_merged.pt',
    '/workspace/a3_train_runs/a3_lora_r2/checkpoints/weights/step_003000.pt',
    '/workspace/a3_train_runs/a3_lora_r3/step_003000_merged.pt',
    '/workspace/a3_train_runs/a3_lora_r3/checkpoints/weights/step_003000.pt',
    '/workspace/a3_train_runs/a3_lora_r2/config.yaml',
    '/workspace/a3_train_runs/a3_lora_r3/config.yaml',
    '/workspace/FasterWAM/checkpoints/fasterwam_release/libero/step_021700.pt',
    '/workspace/FasterWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth',
    '/workspace/FasterWAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth',
    '/workspace/a3_eval/r3_c0_diag_600/manifest.json',
    '/workspace/a3_eval/r3_c0_diag_600/lift.json',
    '/workspace/a3_eval/r3_c0_diag_600/gripper_21.json',
    '/workspace/a3_eval/light_state_test2/report.json',
    '/workspace/a3_eval/light_state_test2/light/adapter.pt',
    '/workspace/a3_eval/light_state_test2/light/optimizer.pt',
    '/workspace/a3_eval/light_state_test2/light/rng.pt',
    '/workspace/a3_eval/light_state_test2/light/scheduler.pt',
    '/workspace/a3_eval/light_state_test2/light/trainer_state.json',
    '/workspace/fasterwam_a3_bridge/bridge/_next/test_light_state.py',
    '/workspace/fasterwam_a3_bridge/bridge/_next/analyze_lift.py',
    '/workspace/fasterwam_a3_bridge/bridge/_next/analyze_gripper.py',
    '/workspace/fasterwam_a3_bridge/bridge/_next/eval_c0_diag.py',
    '/workspace/_r4_stage/diskinv.sh',
    '/workspace/_r4_stage/mktab.py',
]
for k in KEYS:
    if k in S:
        print('%d\t%s\t%s' % (S[k], k, H[k]))
    else:
        print('MISSING\t%s\t-' % k)
parq = [p for p in S if p.endswith('.parquet')]
print('AGG\t%d parquet files\t%d bytes' % (len(parq), sum(S[p] for p in parq)))
npz = [p for p in S if p.endswith('.npz')]
print('AGG\t%d npz files\t%d bytes' % (len(npz), sum(S[p] for p in npz)))
rest = {p for p in S} - set(parq) - set(npz) - set(KEYS)
print('AGG\t%d other files\t%d bytes' % (len(rest), sum(S[p] for p in rest)))
