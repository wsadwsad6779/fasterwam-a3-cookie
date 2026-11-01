cd /workspace/_r4_stage
cat am0.txt am1.txt am2.txt am3.txt am4.txt am5.txt > am.b64
wc -c am.b64
base64 -d am.b64 > autonomy_metrics.py
sha256sum autonomy_metrics.py
wc -c autonomy_metrics.py
python -m py_compile autonomy_metrics.py && echo PY_OK
