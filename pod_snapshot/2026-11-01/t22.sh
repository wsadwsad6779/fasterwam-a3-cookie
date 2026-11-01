cd /workspace/FasterWAM
grep -nE 'proprio_encoder|trainable|requires_grad|modules_to_save|target_modules|lora' src/fasterwam/models/fastwam.py | head -20
echo ==R4PATCH==
grep -nE 'proprio|trainable|requires_grad|modules_to_save|adapter' src/fasterwam/r4_patch.py | head -25
echo ==CONFIG==
grep -nE 'lora|target_modules|train|proprio' configs/task/a3_fasterwam.yaml | head -30
echo ==COUNT==
grep -nE 'trainable' src/fasterwam/trainer.py | head -10
echo DONE