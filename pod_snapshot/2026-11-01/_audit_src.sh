cd /workspace/FasterWAM || exit 1
{
echo "=== delta mask usage ==="
grep -rn "delta_action_dim_mask" src | head -20
echo "=== warmup ==="
grep -rn -i "warmup" src/fasterwam/trainer.py src/fasterwam/runtime.py | head -20
echo "=== shuffle/dataloader ==="
grep -rn -E "shuffle|DataLoader(" src/fasterwam/trainer.py src/fasterwam/runtime.py | head -20
echo "=== a3_video_dataset norm/action ==="
grep -n -E "normaliz|action|num_frames|windows|pretrained_norm_stats|stepwise" src/fasterwam/datasets/lerobot/a3_video_dataset.py | head -50
echo "=== file size ==="
wc -l src/fasterwam/datasets/lerobot/a3_video_dataset.py src/fasterwam/datasets/lerobot/processors/fastwam_processor.py
echo DONE
} > /workspace/_audit_src.txt 2>&1
