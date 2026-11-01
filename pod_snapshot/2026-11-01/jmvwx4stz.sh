cd /workspace/_r5_stage
head -30 adjudicate_wc.py
echo --------
grep -nE "argv|argparse|json.dump|REFUSE|^def|^ " adjudicate_wc.py | head -40