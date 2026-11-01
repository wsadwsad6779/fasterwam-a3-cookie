grep -rn "lora_named_parameters|adapter_state_dict|save_light_state|load_light_state|save_adapter|load_adapter" --include=*.py /workspace/FasterWAM/src /workspace/fasterwam_a3_bridge /workspace/fasterwam-a3-cookie /workspace/_r4_stage /workspace/_r5_stage 2>/dev/null | grep -v ipynb | head -25
echo TORCH
python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())'
echo VIDEO_R5
ls /workspace/a3_eval/r5_b0_c0_paired | head -20
echo VIDCOUNT
find /workspace/a3_eval/r4_b0_c0_paired /workspace/a3_eval/r5_b0_c0_paired -name '*.mp4' | wc -l
echo FREE
free -g | head -2
df -B1 /workspace | tail -1
echo DONE