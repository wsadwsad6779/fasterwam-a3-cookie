sed -n "33,46p" /workspace/FasterWAM/src/fasterwam/light_state.py
echo LOAD
sed -n "52,60p" /workspace/FasterWAM/src/fasterwam/light_state.py
echo YAML
grep -n -A 4 "raw_shape" /workspace/FasterWAM/configs/data/a3_2cam.yaml | head -14
grep -n proprio_output_dim /workspace/FasterWAM/configs/data/a3_2cam.yaml
echo ARRAYS
ls -l /workspace/a3_data/a3_fw_force/ 2>&1 | head -6
grep -n -B 2 -A 4 "load_adapter(model" /workspace/fasterwam_a3_bridge/bridge/_next/common.py | head -12