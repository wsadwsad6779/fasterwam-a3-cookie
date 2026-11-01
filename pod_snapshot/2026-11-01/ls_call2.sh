cd /workspace
grep -rnE 'lora_named_parameters|adapter_state_dict|save_light_state|load_light_state|save_adapter|load_adapter' --include=*.py FasterWAM/src fasterwam_a3_bridge fasterwam-a3-cookie _r4_stage _r5_stage 2>/dev/null | grep -v ipynb
echo DONE