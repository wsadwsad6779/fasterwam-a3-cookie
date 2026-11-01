cd /workspace/FasterWAM
sed -n '75,100p' src/fasterwam/trainer.py
echo ====
sed -n '275,300p' src/fasterwam/trainer.py
echo ==PROPRIOENC==
find . -name fastwam.py | head -3
grep -rnE 'proprio_encoder' src/ | head -12
echo ==PILOT==
ls -la /workspace/a3_eval/pilot_single_lowforce_clean /workspace/a3_eval/pilot_single_lowforce_noisy 2>/dev/null | head
echo DONE