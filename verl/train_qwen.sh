# Copyright (c) 2025, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/Tool-N1/blob/main/LICENSE

#!/usr/bin/env bash

export N_GPUS=8
export BASE_MODEL="path/to/model/Qwen2.5-7B-Instruct"
export DATA_DIR="path/to/data/tool_call_data_verl"
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND="XFORMERS"

export GPU_UT=0.6
export BA_SIZE=1024
export MAX_PROMPT_LEN=4096
export PRO_NAME="qwen"
export EXPERIMENT_NAME="qwen"
export LOG_DIR="path/to/logs/qwen.txt"

export LR=1e-6
export ENTROPY=0
export MAX_RES=8192
export TEMPERATURE=0.7
export EPOCH=7
export KL_COE=0.001

bash examples/agent/qwen.sh