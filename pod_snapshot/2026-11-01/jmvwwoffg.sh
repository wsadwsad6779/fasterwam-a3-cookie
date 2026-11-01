cd /workspace/FasterWAM
grep -n "def lora_named_parameters|def adapter_state_dict|def save_light_state|def load_light_state|def save_adapter|def load_adapter" src/fasterwam/light_state.py
echo ===
grep -rn --include=*.py proprio_encoder . 2>/dev/null | head -30
echo ===
ls src/fasterwam/datasets src/fasterwam/models src/fasterwam/utils