#!/usr/bin/env bash
set -u
S=/workspace/_r4_stage
sed -e s/seed22/seed21/g -e "s/--seeds 22/--seeds 21/" $S/r4_video.sh > $S/r4_video21.sh
{
  echo "== diff seed22-runner vs seed21-runner"
  diff $S/r4_video.sh $S/r4_video21.sh
  echo "== sha"
  sha256sum $S/r4_video.sh $S/r4_video21.sh
  echo "== launch seed21 video run in background"
  nohup bash $S/r4_video21.sh > /dev/null 2>&1 &
  echo "video21_pid=$!"
  echo "== launch checks in background"
  nohup bash $S/tko_check.sh > /dev/null 2>&1 &
  echo "check_pid=$!"
} > $S/tko_launch.log 2>&1
cat $S/tko_launch.log
