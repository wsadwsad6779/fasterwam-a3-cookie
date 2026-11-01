# Collect reproduction evidence only: commits, working-tree diffs, environment.
D=/workspace/_r6_stage/out/repro
mkdir -p $D
for R in FasterWAM a3_dual_arm_sim fasterwam_a3_bridge fasterwam-a3-cookie; do
  if [ -d /workspace/$R/.git ]; then
    git -C /workspace/$R rev-parse HEAD > $D/${R}_head.txt 2>&1
    git -C /workspace/$R branch --show-current > $D/${R}_branch.txt 2>&1
    git -C /workspace/$R status --porcelain > $D/${R}_status.txt 2>&1
    git -C /workspace/$R log -1 --date=iso-strict --format=%H%n%ci%n%s > $D/${R}_lastcommit.txt 2>&1
    git -C /workspace/$R diff HEAD > $D/${R}_diff.patch 2>&1
    git -C /workspace/$R remote -v > $D/${R}_remote.txt 2>&1
  else
    echo NO_GIT_DIRECTORY > $D/${R}_head.txt
  fi
done
python3 -V > $D/python_version.txt 2>&1
python3 -c 'import torch;print(torch.__version__, torch.version.cuda, torch.cuda.is_available())' > $D/torch_env.txt 2>&1
pip freeze > $D/pip_freeze.txt 2>&1
cat /proc/driver/nvidia/version > $D/nvidia_driver.txt 2>&1
/usr/bin/nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > $D/gpu.txt 2>&1
ls -la /workspace > $D/workspace_listing.txt 2>&1
df -B1 /workspace > $D/df_workspace.txt 2>&1
echo REPRO_DONE
