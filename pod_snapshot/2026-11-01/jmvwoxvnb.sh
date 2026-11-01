set -e
rm -rf /tmp/r5up
mkdir -p /tmp/r5up
tar -xzf /workspace/_r4_stage/r5up.tgz -C /tmp/r5up
find /tmp/r5up -type f | sort | xargs ls -l
cp /tmp/r5up/src/r5_patch.py /workspace/FasterWAM/src/fasterwam/r5_patch.py
cp /tmp/r5up/scripts/train_light_r5.py /workspace/FasterWAM/scripts/train_light_r5.py
cp /tmp/r5up/stage/run_r5.sh /workspace/_r4_stage/run_r5.sh
cp /tmp/r5up/stage/run_r5_dry.sh /workspace/_r4_stage/run_r5_dry.sh
chmod +x /workspace/_r4_stage/run_r5.sh /workspace/_r4_stage/run_r5_dry.sh
cd /workspace/FasterWAM
python -m py_compile src/fasterwam/r5_patch.py src/fasterwam/r4_patch.py scripts/train_light_r5.py scripts/train_light.py && echo PYCOMPILE_OK
bash -n /workspace/_r4_stage/run_r5.sh && bash -n /workspace/_r4_stage/run_r5_dry.sh && echo BASHCHK_OK
sha256sum /workspace/FasterWAM/src/fasterwam/r5_patch.py /workspace/FasterWAM/scripts/train_light_r5.py /workspace/_r4_stage/run_r5.sh /workspace/_r4_stage/run_r5_dry.sh
