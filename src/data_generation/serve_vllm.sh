#!/bin/bash

MODEL_PATH="allenai/olmOCR-7B-0225-preview"

PORTS=(8100)
GPU_GROUPS=("4,5")

for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    gpus="${GPU_GROUPS[$i]}"
    echo "Starting VLLM server on port $port using GPUs $gpus for model $MODEL_PATH..."

    CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_USE_V1=0 \
    VLLM_LOG_LEVEL=info \
    VLLM_DISABLE_HTTP_LOGS=true \
    nohup vllm serve "$MODEL_PATH" \
        --gpu-memory-utilization 0.95 \
        --max-model-len 16384 \
        --tensor-parallel-size 2 \
        --host 0.0.0.0 \
        --port "$port" \
        --served-model-name "olmOCR-7B-${port}" \
        --disable-log-requests > "vllm_${port}.log" 2>&1 &
done

echo "Done"
