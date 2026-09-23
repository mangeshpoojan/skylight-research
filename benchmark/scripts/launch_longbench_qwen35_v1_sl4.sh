#!/bin/bash
# Tune PQ+IS (Sink 4 / Local 4), then run official LongBench v1.
# Configs: dense, pq_is_{2,5}, pqcache_{2,5}, oracle_topk_{2,5}.
# Hybrid Qwen3.5-27B, micro-metrics, THUDM 21 tasks, middle trunc 31500.
set -euo pipefail

export HOME=/data/prithvi/home
mkdir -p "$HOME"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate sparse_attention_hub
else
  export PATH="/data/prithvi/envs/sparse_attention_hub/bin:$PATH"
fi

cd /data/prithvi/skylight-research
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/data/prithvi/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/data/prithvi/hf_cache}"

LOG_DIR=/data/prithvi/pq_importance_runs/longbench_qwen35_27b_v1_sl4/logs
mkdir -p "$LOG_DIR"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
fi

echo "[$(date)] python: $(command -v python)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[$(date)] Sink/Local = 4/4"
nvidia-smi -L || true

echo "[$(date)] Starting PQ+IS Ray Tune (sl4)"
python benchmark/scripts/tune_pq_importance_qwen35_27b_sl4.py \
  2>&1 | tee "$LOG_DIR/tune_pq_is.log"
echo "[$(date)] Ray Tune finished"

CONFIGS=(dense pq_is_2 pq_is_5 pqcache_2 pqcache_5 oracle_topk_2 oracle_topk_5)
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"

echo "[$(date)] Launching LongBench v1 sl4 sweep on ${#CONFIGS[@]} configs"
pids=()
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "[$(date)] GPU $gpu -> $cfg"
  CUDA_VISIBLE_DEVICES="$gpu" python benchmark/scripts/run_longbench_qwen35_sl4.py \
    --config "$cfg" \
    > "$LOG_DIR/${cfg}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  cfg="${CONFIGS[$i]}"
  if wait "$pid"; then
    echo "[$(date)] $cfg finished ok (pid $pid)"
  else
    echo "[$(date)] $cfg FAILED (pid $pid)"
    fail=1
  fi
done

echo "[$(date)] sl4 sweep complete, fail=$fail"
exit "$fail"
