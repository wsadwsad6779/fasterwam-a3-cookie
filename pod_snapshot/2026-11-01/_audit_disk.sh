cd /workspace || exit 1
{
echo "=== data ==="
ls -la data | head -15
du -sh data/* 2>/dev/null | sort -h | tail -8
echo "=== FasterWAM ckpt ==="
du -sh FasterWAM/checkpoints/* 2>/dev/null
ls -la FasterWAM/checkpoints/fasterwam_release/libero/ 2>/dev/null
echo "=== a3_train_runs ==="
du -sh a3_train_runs/*
echo "=== bridge outputs ==="
du -sh fasterwam_a3_bridge/bridge/outputs/* 2>/dev/null
echo "=== source symlink ==="
ls -la a3_data/ | head -12
echo DONE
} > /workspace/_audit_disk.txt 2>&1
