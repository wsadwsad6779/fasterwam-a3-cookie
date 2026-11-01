cd /workspace/FasterWAM
grep -rn --include=*.py -l light_state src scripts examples 2>/dev/null | head -20
echo ===
grep -rn --include=*.py -l lora_named_parameters . 2>/dev/null | head -20
echo ===
ls src/fasterwam | head -50