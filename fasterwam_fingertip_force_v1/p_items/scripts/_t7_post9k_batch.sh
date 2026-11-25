#!/bin/bash
# Minimal 9k guard: when F0 completes 9k, KILL the continuation chain so 12k cannot launch,
# free the single GPU, and HOLD F0 at 9k. No diagnostics here (the real batch is run manually).
set -u
P=/workspace/fwv1/p_items
LOG=/workspace/fwv1/logs/post9k_batch.log
exec >> "$LOG" 2>&1
echo "POST9K_HOLD_WAIT $(date -u +%FT%TZ)"
for i in $(seq 1 540); do [ -f "$P/F0_EVAL_9K_DONE.json" ] && break; sleep 20; done
echo "POST9K_9K_SEEN $(date -u +%FT%TZ)"
sleep 20
for pid in $(pgrep -f "[_]t7_f0_continue_chain.sh"); do echo "POST9K_TERM_CHAIN $pid"; kill -TERM "$pid" 2>/dev/null; done
sleep 3
for pid in $(pgrep -f "[_]t7_f0_continue_chain.sh"); do kill -KILL "$pid" 2>/dev/null; done
for pid in $(pgrep -f "[_]t7_train_f0.py"); do echo "POST9K_TERM_TRAIN $pid"; kill -TERM "$pid" 2>/dev/null; done
for i in $(seq 1 60); do pgrep -f "[_]t7_train_f0.py" >/dev/null || break; sleep 5; done
for pid in $(pgrep -f "[_]t7_train_f0.py"); do kill -KILL "$pid" 2>/dev/null; done
while pgrep -f "[_]t7_eval" >/dev/null; do sleep 10; done
echo "POST9K_GPU_FREE_FOR_MANUAL $(date -u +%FT%TZ)"
echo "{\"status\":\"GPU_FREE_FOR_MANUAL\",\"ts\":\"$(date -u +%FT%TZ)\",\"note\":\"12k killed; F0 held at 9k; manual batch (H-sweep on F2 15k + B2/B5/B3/B4 + R1) runs here\"}" > "$P/GPU_FREE_FOR_MANUAL_BATCH.json"
echo "{\"status\":\"F0_HELD\",\"step\":9000,\"ts\":\"$(date -u +%FT%TZ)\",\"note\":\"12k not started per user; F0 paused at 9k\"}" > "$P/F0_HELD_AT_9K.json"
echo "POST9K_HOLD_DONE $(date -u +%FT%TZ)"
