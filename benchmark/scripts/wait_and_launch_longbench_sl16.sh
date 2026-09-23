#!/bin/bash
# Wait until the 128/128 LongBench v1 sweep leaves the GPUs, then launch sl16.
set -euo pipefail

export HOME=/data/prithvi/home
mkdir -p "$HOME"

LOG_DIR=/data/prithvi/pq_importance_runs/longbench_qwen35_27b_v1_sl16/logs
mkdir -p "$LOG_DIR"
WATCH_LOG="$LOG_DIR/waiter.log"

current_sweep_running() {
  # Character-class trick so this pgrep does not match itself.
  pgrep -f '[p]ython benchmark/scripts/run_longbench_qwen35_27b.py' >/dev/null 2>&1 \
    || pgrep -f '/benchmark/scripts/launch_longbench_qwen35_v1.sh$' >/dev/null 2>&1
}

echo "[$(date)] sl16 waiter started" | tee -a "$WATCH_LOG"
echo "[$(date)] waiting for 128/128 sweep processes to exit" | tee -a "$WATCH_LOG"

while current_sweep_running; do
  n_run=$(pgrep -f '[p]ython benchmark/scripts/run_longbench_qwen35_27b.py' | wc -l || true)
  echo "[$(date)] still running: run_longbench_qwen35_27b.py count=$n_run" | tee -a "$WATCH_LOG"
  sleep 120
done

echo "[$(date)] 128/128 sweep processes gone; sleeping 60s for GPU teardown" | tee -a "$WATCH_LOG"
sleep 60

echo "[$(date)] launching sl16 sweep" | tee -a "$WATCH_LOG"
exec /bin/bash /data/prithvi/skylight-research/benchmark/scripts/launch_longbench_qwen35_v1_sl16.sh
