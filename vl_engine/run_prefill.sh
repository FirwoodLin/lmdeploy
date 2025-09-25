#!/bin/bash

export ROLE='Prefill'

#### hybper parameter begin ############

# Model Setting
# export MODEL_PATH='/mnt/137_nvme2/huggingface_hub/hub/models--deepseek-ai--DeepSeek-V3/snapshots/86518964eaef84e3fdd98e9861759a1384f9c29d'
export MODEL_PATH="/mnt/137_nvme3/interns1/InternS1-235b-rc23-fp8-remote"

# distributed setting
export NODE_RANK=$1
export GPU_NUMS=8
export MASTER_ADDR='10.130.8.143'
export MASTER_PORT=29555

# proxy setting
export PROXY_URL='10.130.8.143:8000'

# batch setting
export MAX_BATCH_SIZE='128'
export MAX_PREFILL_TOKEN_NUM='4096'

# cache setting
export CACHE_MAX_ENTRY_COUNT='0.40'

# TBO setting
export ENABLE_MICROBATCH="--enable-microbatch"
# export ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD='8190'

# Memory Pool Setting
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False'

# log setting
export server_start_time=`date '+%Y-%m-%d_%H-%M'`
export LOG_LEVEL=ERROR
export LOG_DIR="2p4d_${ROLE}_${MAX_BATCH_SIZE}_${CACHE_MAX_ENTRY_COUNT}_${server_start_time}_${MODEL_PATH}"
mkdir -p ${LOG_DIR}

#### hybper parameter end ############

export USER=root

ray stop --force
ray start --head --port 6677 --disable-usage-stats
sleep 2

export NNODES=1
export LMDEPLOY_FAKE_EPLB=TRUE
export TRANSFORMERS_OFFLINE=1
export LMDEPLOY_DP_MASTER_ADDR=${MASTER_ADDR}
export LMDEPLOY_DP_MASTER_PORT=29500

export SLIME_LOG_LEVEL=1
export SLIME_SERVICE_LEVEL=2

export HOME=/root/

export DG_JIT_CACHE_DIR=/root/

# Print all parameters before starting
echo "========== Configuration Parameters =========="
echo "ROLE: $ROLE"
echo "NODE_RANK: $NODE_RANK"
echo "GPU_NUMS: $GPU_NUMS"
echo "PROXY_URL: $PROXY_URL"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "MAX_BATCH_SIZE: $MAX_BATCH_SIZE"
echo "MAX_PREFILL_TOKEN_NUM: $MAX_PREFILL_TOKEN_NUM"
echo "CACHE_MAX_ENTRY_COUNT: $CACHE_MAX_ENTRY_COUNT"
echo "DEEPEP_MAX_BATCH_SIZE: $DEEPEP_MAX_BATCH_SIZE"
echo "ENABLE_MICROBATCH: $ENABLE_MICROBATCH"
echo "ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD: $ENABLE_MICROBATCH_PREFILL_TOKEN_THRESHOLD"
echo "ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD: $ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD"
echo "ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD: $ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD"
echo "USER: $USER"
echo "MODEL_PATH: $MODEL_PATH"
echo "LMDEPLOY_DP_MASTER_ADDR: $LMDEPLOY_DP_MASTER_ADDR"
echo "LMDEPLOY_DP_MASTER_PORT: $LMDEPLOY_DP_MASTER_PORT"
echo "PYTORCH_CUDA_ALLOC_CONF: $PYTORCH_CUDA_ALLOC_CONF"
echo "DG_JIT_CACHE_DIR: $DG_JIT_CACHE_DIR"
echo "HOME: $HOME"
echo "TRANSFORMERS_OFFLINE: $TRANSFORMERS_OFFLINE"
echo "LMDEPLOY_FAKE_EPLB: $LMDEPLOY_FAKE_EPLB"
echo "=============================================="

env \
TRANSFORMERS_OFFLINE=1 \
LMDEPLOY_FAKE_EPLB=TRUE \
DG_JIT_CACHE_DIR=/root/ \
DEEPEP_MAX_BATCH_SIZE='64' \
ENABLE_MICROBATCH_PREFILL_BATCHSIZE_THRESHOLD='4' \
ENABLE_MICROBATCH_DECODE_BATCHSIZE_THRESHOLD='256' \
python -m lmdeploy serve api_server \
    ${MODEL_PATH}                                    \
    --backend pytorch                                \
    --ep ${GPU_NUMS}                                 \
    --dp ${GPU_NUMS}                                 \
    --proxy-url http://${PROXY_URL}                  \
    --nnodes ${NNODES}                               \
    --cache-max-entry-count ${CACHE_MAX_ENTRY_COUNT} \
    --max-prefill-token-num ${MAX_PREFILL_TOKEN_NUM} \
    --role ${ROLE}                                   \
    --node-rank ${NODE_RANK} ${ENABLE_MICROBATCH} --max-batch-size ${MAX_BATCH_SIZE} --model-format fp8 --log-level ${LOG_LEVEL} --enable-metrics 2>&1 | tee ${LOG_DIR}/dp${GPU_NUMS}ep${GPU_NUMS}_${ROLE}_node${NODE_RANK}.log
