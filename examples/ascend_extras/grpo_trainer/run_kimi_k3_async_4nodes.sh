#!/usr/bin/env bash
set -uo pipefail
export ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE=64
export ROLLOUT_CUDAGRAPH_CAPTURE_SIZES='[1,2,4,8,16,32,64]'

# ---- 后端根：代码软链到容器 /verl /FSDPTurbo /vllm /vllm-ascend，数据落 NFS ----
export BACKEND_ROOT=/mnt/share/r00947109/kimi-k3-megatron
# ---- 剪枝权重（多机 separate_async 使用） ----
export MODEL_PATH=/mnt/share/w00848461/weights/K3-top16-pruned_bf16_countbench_128_n4
# export MODEL_PATH=${MODEL_PATH:-/mnt/share/r00947109/kimi-k3-megatron/weight}
# ---- 4 个节点 IP ----
export TRAINER_NNODES=2
export ROLLOUT_NNODES=2
export VERL_EXPECTED_NODE_IPS="${VERL_EXPECTED_NODE_IPS:-80.5.25.124,80.5.25.128,80.5.25.130,80.5.25.131}"

# ---- 多机 separate_async 训练参数 ----
export NPUS_PER_NODE=16
export SPLIT_TRAIN_GPUS=16
export SPLIT_ROLLOUT_GPUS=16
export TURBO_EP_SIZE=16
# export ROLLOUT_TP=16
# export ROLLOUT_TP=4 # previous setting
export ROLLOUT_TP=16
export ROLLOUT_EP=16
export TRAIN_BATCH_SIZE=16
export PPO_MINI_BATCH_SIZE=16
export ROLLOUT_N=4
export VERL_VLLM_MULTIMODAL_TOKEN_MARGIN=8192
export MAX_PROMPT_LENGTH=1024
export MAX_RESPONSE_LENGTH=512
# export MAX_RESPONSE_LENGTH=256
# export MAX_PROMPT_LENGTH=256
# export VERL_VLLM_MAX_NUM_SEQS=16
# export ROLLOUT_GPU_MEMORY_UTILIZATION=0.9
# export VERL_VLLM_MAX_NUM_SEQS=8 # previous setting
export VERL_VLLM_MAX_NUM_SEQS=64
export VERL_VLLM_MAX_NUM_BATCHED_TOKENS=4096
export VERL_TOTAL_TRAINING_STEPS=10
export VERL_SAVE_FREQ=-1
export VERL_RESUME_MODE=disable
# export VERL_VLLM_ENFORCE_EAGER=False
# export VERL_VLLM_ENFORCE_EAGER=True # before TP4 graph test
export VERL_VLLM_ENFORCE_EAGER=False
# export ROLLOUT_ENFORCE_EAGER=True # before TP4 graph test
export ROLLOUT_ENFORCE_EAGER=False
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.8
export VERL_EXPERIMENT_NAME=kimi_k3_20261031_seqs64_profile_actor_only080_exp1_attempt31

export PARAMETER_SYNC_STEP=1

export ROLLOUT_DP=1

export ACTOR_MICRO_BATCH_SIZE_PER_GPU=1

export LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=2

D=/mnt/share/r00947109/kimi-k3-megatron/verl/examples/ascend_extras/grpo_trainer
LOG_DIR=/mnt/share/r00947109/kimi-k3-megatron/verl/logs
mkdir -p "$LOG_DIR"
NODE="$(hostname -s 2>/dev/null || hostname)"
LOG="$LOG_DIR/kimi_k3_fsdpturbo_separate_async_4nodes_$(date +%Y%m%d_%H%M%S)_${NODE}.log"

# Record device memory separately from training/progress output.
NPU_MEMORY_MONITOR_PYTHON="${NPU_MEMORY_MONITOR_PYTHON:-/usr/local/python3.12.13/bin/python3}"
NPU_MEMORY_LOG="${LOG%.log}.npu_memory.csv"
"$NPU_MEMORY_MONITOR_PYTHON" "$D/monitor_npu_memory.py" \
  --output "$NPU_MEMORY_LOG" --node "${NODE_IP:-$NODE}" \
  --interval 30 --parent-pid "$$" \
  2> "${LOG%.log}.npu_memory.errors.log" &
NPU_MEMORY_MONITOR_PID=$!
stop_npu_memory_monitor() {
  kill "$NPU_MEMORY_MONITOR_PID" 2>/dev/null || true
  wait "$NPU_MEMORY_MONITOR_PID" 2>/dev/null || true
}
trap stop_npu_memory_monitor EXIT

export ACTOR_PARAM_OFFLOAD=False
export ACTOR_OPTIMIZER_OFFLOAD=False
bash "$D/ray_start_multi_nodes_fsdpturbo_separate_async.sh" \
  "$@" 2>&1 | tee "$LOG"
