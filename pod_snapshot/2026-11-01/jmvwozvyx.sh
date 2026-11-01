echo '--- collect scripts ---'
find /workspace -maxdepth 2 -type d -not -path '*/.*' | sort
echo '--- who writes arrays.npz ---'
grep -rln --include=*.py 'arrays.npz' /workspace 2>/dev/null | grep -v site-packages | head -20
echo '--- pad_force sources ---'
grep -rln --include=*.py -e 'pad_force' -e 'sensor_force' /workspace 2>/dev/null | grep -v site-packages | head -20
