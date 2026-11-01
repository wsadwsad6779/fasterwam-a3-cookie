set -e
cd /workspace/_r4_stage
for i in 0 1 2 3 4; do printf '%s %s\n' \"$i\" \"$(wc -c < c$i.txt)\"; done
cat c0.txt c1.txt c2.txt c3.txt c4.txt > r5up.b64
wc -c r5up.b64
base64 -d r5up.b64 > r5up.tgz
sha256sum r5up.tgz
ls -l r5up.tgz
