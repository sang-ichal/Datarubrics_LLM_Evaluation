#!/bin/bash

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
MODEL_PATH="/data1/models/Qwen/Qwen3-32B"

PORTS=(12321) 
GPU_GROUPS=("0,1,2,3") 

LOG_DIR="./vllm_logs"
mkdir -p "$LOG_DIR"

for i in "${!PORTS[@]}"; do
    port="${PORTS[$i]}"
    gpus="${GPU_GROUPS[$i]}"
    # Dynamically determine tensor_parallel_size based on the number of GPUs in the group
    num_gpus=$(echo "$gpus" | awk -F, '{print NF}')

    served_model_name="qwen3-32b-${port}"

    echo "Starting VLLM server on port $port using GPUs $gpus for model $MODEL_PATH..."
    echo "Served model name will be: $served_model_name"
    echo "Tensor parallel size: $num_gpus"

    CUDA_VISIBLE_DEVICES="$gpus" \
    VLLM_USE_V1=0 \
    VLLM_LOG_LEVEL=info \
    VLLM_DISABLE_HTTP_LOGS=true \
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --tensor-parallel-size "$num_gpus" \
        --host 0.0.0.0 \
        --port "$port" \
        --served-model-name "$served_model_name" \
        --disable-log-requests > "${LOG_DIR}/vllm_${served_model_name}.log" 2>&1 &
done

echo "Done"