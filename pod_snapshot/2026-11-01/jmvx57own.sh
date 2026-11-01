cat /workspace/ls_call2.out
echo RM_BUILD
grep -nE 'import|build|pretrained|apply_lora|load_adapter|cuda|dtype' /workspace/fasterwam_a3_bridge/bridge/_next/run_matrix.py | head -30