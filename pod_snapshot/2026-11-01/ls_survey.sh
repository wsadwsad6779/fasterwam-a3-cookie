wc -c /workspace/FasterWAM/src/fasterwam/light_state.py
echo TRAINABLE_LOG
grep -n trainable /workspace/a3_train_runs/a3_lora_r5/train.log | head -3
echo PROPRIO_DEF
grep -rn proprio_encoder /workspace/FasterWAM/src/fasterwam/ | head -20
echo AUDIT_SCRIPTS
grep -rln requires_grad /workspace/_r4_stage /workspace/_r5_stage 2>/dev/null | head
echo BRIDGE_NEXT
ls /workspace/_r4_stage/bridge_next/ 2>/dev/null | head
echo DONE