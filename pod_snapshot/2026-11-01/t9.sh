cd /workspace/a3_dual_arm_sim
echo ==ADAPTERS==
sed -n '160,230p' src/a3_dual_arm_sim/workflows/policy_adapters.py
echo ==BOXSUPPORT==
sed -n '1,30p' src/a3_dual_arm_sim/controllers/box_support.py
echo ==2X10SETTINGS==
grep -nE 'first_grasp|required_cookies|episodes|seed' configs/cookie_2x10.yaml | head -20
echo ==DRYRUN==
timeout 60 python examples/collect_cookie_2x10.py --dry-run --first-grasp 1 --episodes 2 --root /workspace/a3_data/_pilot_check 2>&1 | tail -25
echo DONE