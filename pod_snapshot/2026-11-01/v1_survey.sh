cd /workspace
find . -maxdepth 5 -name light_state.py -o -maxdepth 5 -name 'a3_video_dataset*' -o -maxdepth 5 -name 'prepare_a3_arrays*' | grep -v ipynb
echo GREP_BUILD
grep -rn 'def build_proprio' --include=*.py . 2>/dev/null | head -5
echo GREP_CLASS
grep -rn 'class A3VideoDataset' --include=*.py . 2>/dev/null | head -3
echo DONE