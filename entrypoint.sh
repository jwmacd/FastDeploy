#!/bin/bash

# Set default values - optimized for 2-bit model on single GPU
MODEL=${MODEL:-"baidu/ERNIE-4.5-300B-A47B-2Bits-Paddle"}
PORT=${PORT:-8001}
WORKER_QUEUE_PORT=${WORKER_QUEUE_PORT:-8181}
CACHE_QUEUE_PORT=${CACHE_QUEUE_PORT:-8182}
METRICS_PORT=${METRICS_PORT:-8182}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
QUANTIZATION=${QUANTIZATION:-"wint2"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-16}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.95}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-"true"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-2048}

# Optional parameters
ENABLE_MM=${ENABLE_MM:-""}
MM_PROCESSOR_KWARGS=${MM_PROCESSOR_KWARGS:-""}
LIMIT_MM_PER_PROMPT=${LIMIT_MM_PER_PROMPT:-""}
REASONING_PARSER=${REASONING_PARSER:-""}

# Set ulimits
ulimit -n 65536
ulimit -u unlimited

# Build command
CMD="python -m fastdeploy.entrypoints.openai.api_server"
CMD="$CMD --model $MODEL"
CMD="$CMD --port $PORT"
CMD="$CMD --engine-worker-queue-port $WORKER_QUEUE_PORT"
CMD="$CMD --cache-queue-port $CACHE_QUEUE_PORT"
CMD="$CMD --metrics-port $METRICS_PORT"
CMD="$CMD --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
CMD="$CMD --quantization $QUANTIZATION"
CMD="$CMD --max-model-len $MAX_MODEL_LEN"
CMD="$CMD --max-num-seqs $MAX_NUM_SEQS"
CMD="$CMD --gpu-memory-utilization $GPU_MEMORY_UTILIZATION"
CMD="$CMD --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS"

# Add chunked prefill if enabled
if [ "$ENABLE_CHUNKED_PREFILL" = "true" ]; then
    CMD="$CMD --enable-chunked-prefill"
fi

# Add optional parameters if set
if [ -n "$ENABLE_MM" ]; then
    CMD="$CMD --enable-mm"
fi

if [ -n "$MM_PROCESSOR_KWARGS" ]; then
    CMD="$CMD --mm-processor-kwargs '$MM_PROCESSOR_KWARGS'"
fi

if [ -n "$LIMIT_MM_PER_PROMPT" ]; then
    CMD="$CMD --limit-mm-per-prompt '$LIMIT_MM_PER_PROMPT'"
fi

if [ -n "$REASONING_PARSER" ]; then
    CMD="$CMD --reasoning-parser $REASONING_PARSER"
fi

# Add any additional arguments passed to the script
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
fi

echo "Starting FastDeploy with command:"
echo "$CMD"
echo ""

# Execute the command
exec $CMD