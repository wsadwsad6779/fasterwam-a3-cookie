#!/bin/bash
set -e
S=/workspace/fasterwam_a3_bridge/bridge/_next
R=/workspace/fasterwam-a3-cookie
for f in dagger_recover.py dagger_to_dataset.py check_a3_dagger_dataset.py; do
  sed "s/\r$//" "$S/$f" > "$R/bridge/_next/$f"
done
sed "s/\r$//" "$R/bridge/_next/README.md" > /tmp/readme.lf
cat /workspace/dagger_section.md >> /tmp/readme.lf
cp /tmp/readme.lf "$R/bridge/_next/README.md"
cp /tmp/readme.lf "$S/README.md"
G="git -c safe.directory=$R"
cd $R
$G add bridge/_next/dagger_recover.py bridge/_next/dagger_to_dataset.py bridge/_next/check_a3_dagger_dataset.py bridge/_next/README.md
$G diff --cached --stat | tail -6
echo SYNC_OK
