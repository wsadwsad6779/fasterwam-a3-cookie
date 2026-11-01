set -u
cd /workspace/_r4_stage
base64 -d r5up.tgz.b64 > r5up.tgz
sha256sum r5up.tgz
tar -xzf r5up.tgz -C /tmp/r5up 2>/dev/null || { mkdir -p /tmp/r5up; tar -xzf r5up.tgz -C /tmp/r5up; }
find /tmp/r5up -type f | sort
echo ==EXPECT f76709d2f5c2cb20ff0c5d45b71bcd27783a447b3abb44462e93ba83ac617220