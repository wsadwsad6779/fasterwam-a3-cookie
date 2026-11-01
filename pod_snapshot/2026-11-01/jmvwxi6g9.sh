head -6 /workspace/adjv2.out
echo == TALLIES
grep -E .dir. /workspace/adjv2.out
echo == WROTE
grep -E WROTE /workspace/adjv2.out
echo == C3
grep -E TALLY /workspace/adjv2.out
echo == END
tail -4 /workspace/adjv2.out