#!/bin/bash
# Tune PQ+IS on Qwen3.5-27B, then run the official LongBench v1 sweep.
# Dense + OracleTopK 20/10/5/2% + PQ+IS 5/2%, hybrid, micro-metrics,
# THUDM/LongBench 21 tasks, middle truncation at 31500.
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

LOG_DIR=/data/prithvi/pq_importance_runs/longbench_qwen35_27b_v1/logs
mkdir -p "$LOG_DIR"

# Use every visible GPU. Default to 0-6 if the caller did not set the mask.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
fi

echo "[$(date)] python: $(command -v python)"
echo "[$(date)] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L || true

echo "[$(date)] Starting PQ+IS Ray Tune"
python benchmark/scripts/tune_pq_importance_qwen35_27b.py \
  2>&1 | tee "$LOG_DIR/tune_pq_is.log"
echo "[$(date)] Ray Tune finished"

CONFIGS=(dense oracle_topk_20 oracle_topk_10 oracle_topk_5 oracle_topk_2 pq_is_5 pq_is_2)
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"

echo "[$(date)] Launching LongBench v1 sweep on ${#CONFIGS[@]} configs"
pids=()
for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  echo "[$(date)] GPU $gpu -> $cfg"
  CUDA_VISIBLE_DEVICES="$gpu" python benchmark/scripts/run_longbench_qwen35_27b.py \
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

echo "[$(date)] sweep complete, fail=$fail"
exit "$fail"
