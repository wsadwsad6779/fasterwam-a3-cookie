free -g | head -3
pgrep -af rd26 | head -3
ps -o pid,rss,cmd -p $(pgrep -f 'python3 -' | head -1) 2>/dev/null | tail -2
echo done