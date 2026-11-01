date -u
for t in wc4 wc6fix cc4 cc5; do echo == $t; grep -c "^.seed." /workspace/$t.out; done
echo WC6FIXTAIL
tail -c 300 /workspace/wc6fix.out
echo CC4
tail -c 200 /workspace/cc4.out