#!/bin/bash

# --- Model Configuration ---
MODEL_NAME_SGLANG="Qwen/Qwen3-32B-FP8"
MODEL_ROOT_SGLANG="/data1/models"
MODEL_PATH_SGLANG="$MODEL_ROOT_SGLANG/$MODEL_NAME_SGLANG"

# --- GPU Configuration ---
# Input: array of available nodes (e.g., 1, 2, 3, 4)
# Each node corresponds to a GPU pair for an sglang server instance.
# Node 1 -> GPUs 0,1 for server 1
# Node 2 -> GPUs 2,3 for server 2
# ...
# The number of servers launched will be the number of elements in available_nodes.
# Maximum of 4 servers implies available_nodes should have at most 4 elements
# if each server uses 2 GPUs and you have 8 GPUs in total (0-7).
available_nodes=(1 2 3 4) # Define your available nodes here. E.g., (1 3) for servers on GPUs 0,1 and 4,5

# --- Server Configuration ---
BASE_PORT_SGLANG=8000
TP_SIZE_PER_SGLANG_SERVER=2 # Tensor parallel size for each sglang server instance

# Counter for assigning ports
port_offset=0
num_servers_launched=0

echo "--- SGLang Multi-Server Configuration ---"
echo "Model Path: $MODEL_PATH_SGLANG"
echo "Base Port: $BASE_PORT_SGLANG"
echo "TP Size per Server: $TP_SIZE_PER_SGLANG_SERVER"
echo "Available nodes for server instances: ${available_nodes[@]}"
echo "-----------------------------------------"

for node_num in "${available_nodes[@]}"; do
    # Calculate GPU indices for this server instance
    # Node 1 uses GPUs 0,1 ( (1-1)*2 and (1-1)*2 + 1 )
    # Node 2 uses GPUs 2,3 ( (2-1)*2 and (2-1)*2 + 1 )
    start_gpu=$(( (node_num - 1) * TP_SIZE_PER_SGLANG_SERVER ))
    end_gpu=$(( start_gpu + TP_SIZE_PER_SGLANG_SERVER - 1 ))
    visible_devices_for_server=$(seq -s, $start_gpu $end_gpu)

    # Assign port sequentially
    current_port=$(( BASE_PORT_SGLANG + port_offset ))
    port_offset=$(( port_offset + 1 ))

    # Construct a unique served model name if needed, or keep it generic
    # For simplicity, keeping it generic, but you could append port or node_num
    served_model_name_instance="qwen3-32b"
    # Example for unique name: served_model_name_instance="${MODEL_NAME_SGLANG}-node${node_num}"

    echo "Preparing SGLang server instance for node $node_num:"
    echo "  GPUs: $visible_devices_for_server"
    echo "  Port: $current_port"
    echo "  Served Model Name: $served_model_name_instance"
    echo "  Log file: sglang_server_node${node_num}_port${current_port}.log"

    CUDA_VISIBLE_DEVICES=$visible_devices_for_server \
    python -m sglang.launch_server \
        --model-path "$MODEL_PATH_SGLANG" \
        --tensor-parallel-size $TP_SIZE_PER_SGLANG_SERVER \
        --quantization fp8 \
        --host 0.0.0.0 \
        --port $current_port \
        --context-length 8192 \
        --served-model-name "$served_model_name_instance" \
        --attention-backend flashinfer \
        > "sglang_server_node${node_num}_port${current_port}.log" 2>&1 &

    echo "Launched SGLang server for node $node_num on port $current_port with GPUs $visible_devices_for_server. PID: $!"
    num_servers_launched=$((num_servers_launched + 1))
    echo "-----------------------------------------"
done

wait # Optional: wait for all background jobs to finish if you need to ensure they've started
echo "Finished launching $num_servers_launched SGLang server(s) based on available_nodes."