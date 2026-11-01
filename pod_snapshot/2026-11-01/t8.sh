cd /workspace/a3_dual_arm_sim
echo ==DEFAULTS==
grep -nE 'DEFAULT_ROOT|DEFAULT_REPO_ID|DEFAULT_CONFIG|^TASK|import' src/a3_dual_arm_sim/data/collect_cookie_2x10_cli.py | head -25
echo ==ARGS_FULL==
sed -n '40,75p' src/a3_dual_arm_sim/data/collect_cookie_2x10_cli.py
echo ==EXAMPLE==
sed -n '1,40p' examples/collect_cookie_2x10.py
echo ==CONFIGS==
find . -name "*.yaml" -path "*config*" | head -20
find . -name "*2x10*" | grep -v pyc | head -20
echo DONE