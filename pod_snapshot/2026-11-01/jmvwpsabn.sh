python3 -c "import a3_dual_arm_sim as m, inspect, os; print(os.path.dirname(inspect.getfile(m)))" 2>&1 | tail -2
ls -d /workspace/*/ | head -30
du -sh /workspace/* 2>/dev/null | sort -h | tail -12