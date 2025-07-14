# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env bash
set -x

export N_GPUS=8
export BASE_MODEL="path/to/model/Qwen2.5-7B-Instruct"
export DATA_DIR="verl/verl/data"
export ROLLOUT_TP_SIZE=2
export VLLM_ATTENTION_BACKEND="XFORMERS"

export GPU_UT=0.6
export BA_SIZE=1024
export MAX_PROMPT_LEN=8192
export PRO_NAME="qwen"
export EXPERIMENT_NAME="qwen"
export LOG_DIR="path/to/logs/qwen.txt"

export LR=1e-6
export ENTROPY=0
export MAX_RES=1024
export TEMPERATURE=0.7
export EPOCH=7
export KL_COE=0.001


python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.filter_overlong_prompts=True \
data.train_files=$DATA_DIR/out_dc_s250610_1.train.parquet \
data.val_files=$DATA_DIR/out_dc_s250610_1.valid.parquet \
data.train_batch_size=$BA_SIZE \
data.max_prompt_length=$MAX_PROMPT_LEN \
data.max_response_length=$MAX_RES \
data.truncation='left' \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=$KL_COE \
actor_rollout_ref.actor.entropy_coeff=$ENTROPY \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.fsdp_config.param_offload=False \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
actor_rollout_ref.actor.use_dynamic_bsz=True \
actor_rollout_ref.actor.ppo_mini_batch_size=64 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
actor_rollout_ref.model.path=$BASE_MODEL \
actor_rollout_ref.model.use_remove_padding=True \
actor_rollout_ref.model.enable_gradient_checkpointing=True \
actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP_SIZE \
actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_UT \
actor_rollout_ref.rollout.temperature=$TEMPERATURE \
actor_rollout_ref.rollout.n=5 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.kl_ctrl.kl_coef=$KL_COE \
trainer.critic_warmup=0 \
trainer.logger=['wandb'] \
+trainer.val_before_train=False \
+actor_rollout_ref.actor.fsdp_config.grad_offload=False \
trainer.default_hdfs_dir=null \
trainer.n_gpus_per_node=$N_GPUS \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.project_name=$PRO_NAME \
trainer.experiment_name=$EXPERIMENT_NAME \
trainer.default_local_dir=$SAVE_DIR \
trainer.total_epochs=$EPOCH 2>&1 | tee -a "${LOG_DIR}"