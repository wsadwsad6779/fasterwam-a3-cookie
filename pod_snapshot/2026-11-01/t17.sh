find /workspace -maxdepth 4 -name env_factory.py 2>/dev/null | head
for f in $(find /workspace -maxdepth 4 -name env_factory.py 2>/dev/null | head -3); do echo "-- $f"; grep -nE 'DEFAULT_CONFIG|required_cookies|task_config|CookieTransferTaskConfig|load_config|half' $f | head -20; done
echo DONE