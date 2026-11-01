base64 -w 120 /workspace/_r5_stage/replay_forces.py > /workspace/_r5_stage/rf.b64
wc -c /workspace/_r5_stage/rf.b64
sha256sum /workspace/_r5_stage/replay_forces.py
awk "/class SingleCookieLowForce/,/^def main/" /workspace/_r5_stage/replay_forces.py | head -70
