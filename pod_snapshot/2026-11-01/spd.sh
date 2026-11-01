URL=https://hf-mirror.com/wsadwsad6779/fasterwam-a3-cookie-lora/resolve/main/step_002000_merged.pt
: > /workspace/spd.txt
for i in 0 1 2 3 4 5 6 7; do (curl -m 12 -s -r $((i*20000000))-$((i*20000000+19999999)) -o /dev/null -w "par$i %{speed_download}
" "$URL" >> /workspace/spd.txt) & done
wait
curl -m 12 -s --noproxy '*' -r 0-20000000 -o /dev/null -w "direct %{speed_download}
" "$URL" >> /workspace/spd.txt 2>&1
