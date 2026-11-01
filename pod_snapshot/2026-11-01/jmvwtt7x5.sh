date; wc -l /workspace/cc1.out; grep -o '"seed": [0-9]*' /workspace/cc1.out | tr '
' ' '; echo; grep -o 'stop": "[a-z_]*' /workspace/cc1.out | tr '
' ' '; ps -eo etime,cmd | grep replay_correct | grep -v grep | head -2