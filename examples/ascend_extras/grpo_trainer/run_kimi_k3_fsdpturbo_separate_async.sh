#!/usr/bin/env bash
# Kimi-K3 CountBench GRPO: vLLM-Ascend rollout + FSDP-Turbo actor/ref.
# Full-async mode (separate_async) with NCCL checkpoint engine for weight sync.
#
# Based on run_kimi_k3_megatron_separate_async.sh (async architecture) and
# run_kimi_k3_fsdpturbo.sh (FSDP-Turbo backend config).
#
# Usage:
#   # Single-node (8 cards: 4 train + 4 rollout)
#   bash run_kimi_k3_fsdpturbo_separate_async.sh
#
#   # Multi-node (started via ray_start_multi_nodes_fsdpturbo_separate_async.sh)
#   # Each node uses 16 NPUs (8 train + 8 rollout) by default.
#   # Run on head node:
#   bash ray_start_multi_nodes_fsdpturbo_separate_async.sh
#   # Run on each worker node:
#   NODE_IP=<worker_ip> bash ray_start_multi_nodes_fsdpturbo_separate_async.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
echo "SCRIPT_DIR=$SCRIPT_DIR"
source "$SCRIPT_DIR/kimi_k3_runtime.sh"
BACKEND_ROOT=${BACKEND_ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)}
echo "BACKEND_ROOT=$BACKEND_ROOT"
PROJECT_ROOT=$BACKEND_ROOT
RUN_MODE=${KIMI_RUN_MODE:-single}
RUN_TIMESTAMP=${VERL_RUN_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}
project_name=${VERL_PROJECT_NAME:-GRPO-Kimi-K3-A3B}

TRAIN_FILE=${TRAIN_FILE:-/mnt/share/w00848461/datasets/countbenchqa_lite/train_448.parquet}
VAL_FILE=${VAL_FILE:-$TRAIN_FILE}

# FSDP-Turbo paths
fsdp_turbo_path="${PROJECT_ROOT}/FSDPTurbo"
verl_path="${PROJECT_ROOT}/verl"
vllm_path="${PROJECT_ROOT}/vllm"
vllm_ascend_path="${PROJECT_ROOT}/vllm-ascend"
runtime_site="${PROJECT_ROOT}/.runtime-site"
cann_python_site=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages

# Model path (FSDP-Turbo needs its own prepared model copy)
# model_source=${/mnt/share/r00947109/kimi-k3-megatron/weight}
model_source=${MODEL_PATH:-/mnt/share/w00848461/weights/K3-top16-pruned_bf16_countbench_128_n4}
model_path="${PROJECT_ROOT}/models/fsdpturbo-$(basename "$model_source")"

# ── Separate async mode: GPU split ──────────────────────────────────────
# In separate_async mode, the trainer (actor/ref) and rollout server run as
# independent Ray resource pools on the same node.  The default split dedicates
# SPLIT_TRAIN_GPUS to the trainer and SPLIT_ROLLOUT_GPUS to the standalone
# rollout server.
SPLIT_TRAIN_GPUS=${SPLIT_TRAIN_GPUS:-4}
SPLIT_ROLLOUT_GPUS=${SPLIT_ROLLOUT_GPUS:-4}

case "$RUN_MODE" in
    single)
        TRAINER_NNODES=${TRAINER_NNODES:-$NNODES}
        ROLLOUT_NNODES=${ROLLOUT_NNODES:-$NNODES}
        NNODES=1
        NPUS_PER_NODE=$((SPLIT_TRAIN_GPUS + SPLIT_ROLLOUT_GPUS))
        # FSDP-Turbo: EP = train GPUs, eFSDP = 1
        TURBO_EP_SIZE=${TURBO_EP_SIZE:-$SPLIT_TRAIN_GPUS}
        ROLLOUT_TP=${ROLLOUT_TP:-$SPLIT_ROLLOUT_GPUS}
        ROLLOUT_EP=${ROLLOUT_EP:-$ROLLOUT_TP}
        VISIBLE_DEVICES_DEFAULT=0,1,2,3,4,5,6,7
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi_k3_fsdpturbo_separate_async_1node_${RUN_TIMESTAMP}}
        ;;
    multinode)
        : "${NNODES:?NNODES must be set by the multi-node Ray entry}"
        if ((NNODES < 2)); then
            echo "multinode mode requires NNODES >= 2, got $NNODES" >&2
            exit 2
        fi
        # Multinode: use full 16 NPUs per node (8 train + 8 rollout).
        # The top-level defaults (4+4) are for single-node; override here.
        SPLIT_TRAIN_GPUS=${SPLIT_TRAIN_GPUS:-8}
        SPLIT_ROLLOUT_GPUS=${SPLIT_ROLLOUT_GPUS:-8}
        # In separate_async multinode mode, trainer and rollout are on different nodes.
        # Each node has NPUS_PER_NODE physical NPUs (set by wrapper). Don't override.
        : "${NPUS_PER_NODE:=$((SPLIT_TRAIN_GPUS > SPLIT_ROLLOUT_GPUS ? SPLIT_TRAIN_GPUS : SPLIT_ROLLOUT_GPUS))}"
        # In separate_async mode, each node runs its own rollout replica.
        # EP/TP are per-node values, not global — same as the Megatron variant.
        TURBO_EP_SIZE=${TURBO_EP_SIZE:-$SPLIT_TRAIN_GPUS}
        ROLLOUT_TP=${ROLLOUT_TP:-$SPLIT_ROLLOUT_GPUS}
        ROLLOUT_EP=${ROLLOUT_EP:-$ROLLOUT_TP}
        VISIBLE_DEVICES_DEFAULT=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
        EXPERIMENT_NAME=${EXPERIMENT_NAME:-kimi_k3_fsdpturbo_separate_async_fullmodel_${NNODES}nodes_${RUN_TIMESTAMP}}
        ;;
    *)
        echo "unsupported KIMI_RUN_MODE: $RUN_MODE (expected single or multinode)" >&2
        exit 2
        ;;
esac

# FSDP topology validation
turbo_fsdp_size=$((TRAINER_NNODES * SPLIT_TRAIN_GPUS))
if ((turbo_fsdp_size % TURBO_EP_SIZE != 0)); then
    echo "FSDP requires turbo_fsdp_size($turbo_fsdp_size) divisible by EP($TURBO_EP_SIZE)" >&2
    exit 2
fi
turbo_efsdp_size=$((turbo_fsdp_size / TURBO_EP_SIZE))
rollout_tp_size=$ROLLOUT_TP
rollout_ep_size=$ROLLOUT_EP
if ((ROLLOUT_EP % ROLLOUT_TP != 0)); then
    echo "rollout EP=$ROLLOUT_EP must be divisible by TP=$ROLLOUT_TP" >&2
    exit 2
fi
ROLLOUT_DP=${ROLLOUT_DP:-$((ROLLOUT_EP / ROLLOUT_TP))}
rollout_world_size=$((ROLLOUT_TP * ROLLOUT_DP))
if ((ROLLOUT_EP != rollout_world_size || SPLIT_ROLLOUT_GPUS % rollout_world_size != 0)); then
    echo "invalid rollout topology: rollout_gpus=$SPLIT_ROLLOUT_GPUS TP=$ROLLOUT_TP DP=$ROLLOUT_DP EP=$ROLLOUT_EP" >&2
    exit 2
fi

# Mode-dependent defaults
if [[ "$RUN_MODE" == single ]]; then
    ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
    ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
    ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
    ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
    TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-4}
else
    ACTOR_MICRO_BATCH_SIZE_PER_GPU=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-1}
    LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}
    ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
    ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
    ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}
    TQ_STORAGE_UNITS=${TQ_STORAGE_UNITS:-$((TRAINER_NNODES * 8))}
fi

TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT=${TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT:-600}
TQ_NUM_THREADS=${TQ_NUM_THREADS:-8}
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES:-256}

ROLLOUT_CUDAGRAPH_MODE=${ROLLOUT_CUDAGRAPH_MODE:-FULL_DECODE_ONLY}
ROLLOUT_CUDAGRAPH_CAPTURE_SIZES=${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-'[1,2,4,8,16]'}
ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE=${ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE:-16}
ROLLOUT_SLEEP_EXTRA_CLEANUP=${ROLLOUT_SLEEP_EXTRA_CLEANUP:-False}

ROLLOUT_N=${ROLLOUT_N:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-512}
rollout_multimodal_token_margin=${VERL_VLLM_MULTIMODAL_TOKEN_MARGIN:-16384}
MAX_MODEL_LENGTH=${MAX_MODEL_LENGTH:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + rollout_multimodal_token_margin))}
ppo_max_token_len=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)))

rollout_max_num_seqs=${VERL_VLLM_MAX_NUM_SEQS:-32}
rollout_max_num_batched_tokens=${VERL_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}
total_training_steps=${VERL_TOTAL_TRAINING_STEPS:-200}
save_freq=${VERL_SAVE_FREQ:-2}
resume_mode=${VERL_RESUME_MODE:-auto}
resume_from_path=${VERL_RESUME_FROM_PATH:-}
max_actor_ckpt_to_keep=${VERL_MAX_ACTOR_CKPT_TO_KEEP:-1}
test_file=${TEST_FILE:-$TRAIN_FILE}

# Separate async requires: train_batch_size == parameter_sync_step * ppo_mini_batch_size
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-1}

PROJECT_NAME=${PROJECT_NAME:-verl_kimi_k3}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/verl/outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}}
LOG_DIR=${LOG_DIR:-${PROJECT_ROOT}/verl/logs}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-10}

# Separate async invariant: train_batch_size == parameter_sync_step * ppo_mini_batch_size
if ((TRAIN_BATCH_SIZE != PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE)); then
    echo "separate_async requires train_batch_size($TRAIN_BATCH_SIZE) == parameter_sync_step($PARAMETER_SYNC_STEP) * ppo_mini_batch_size($PPO_MINI_BATCH_SIZE)" >&2
    exit 2
fi

# Prepare FSDP-Turbo model copy (copy Python code from FSDP-Turbo into model dir)
python3 "$SCRIPT_DIR/prepare_kimi_k3_model.py" "$model_source" "$model_path" --code "$fsdp_turbo_path/fsdp_turbo/models/kimi"

# ── Environment ─────────────────────────────────────────────────────────
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$VISIBLE_DEVICES_DEFAULT}
export NON_MEGATRON=true
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT
export TQ_NUM_THREADS
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-0}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export FSDP_TURBO_ROOT="${fsdp_turbo_path}"
export VLLM_ASCEND_PATH="${vllm_ascend_path}"
export PYTHONPATH="${KIMI_EXTRA_PYTHONPATH:+${KIMI_EXTRA_PYTHONPATH}:}${PROJECT_ROOT}/.python_deps/modelopt-0.46.0:${runtime_site}:${verl_path}:${vllm_path}:${vllm_ascend_path}:${fsdp_turbo_path}:${fsdp_turbo_path}/examples:${cann_python_site}"
export VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARN}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-WARN}
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-1}
export VLLM_BATCH_INVARIANT=${VLLM_BATCH_INVARIANT:-0}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-true}
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_VERSION=${VLLM_VERSION:-0.26.0}
export VLLM_ASCEND_ENABLE_FLASHCOMM=${VLLM_ASCEND_ENABLE_FLASHCOMM:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
export VERL_KIMI_MLP_CHUNK_TOKENS=${VERL_KIMI_MLP_CHUNK_TOKENS:-128}

# Performance optimizations
export VLLM_ASCEND_TASK_QUEUE_ENABLE=1
export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=${MULTI_STREAM_MEMORY_REUSE:-1}
export CPU_AFFINITY_CONF=${CPU_AFFINITY_CONF:-1}
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export HCCL_SOCKET_IFNAME=${HCCL_SOCKET_IFNAME:-enp48s3u1u1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-${HCCL_SOCKET_IFNAME}}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-auto}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-auto}
export HCCL_IF_BASE_PORT=${HCCL_IF_BASE_PORT:-50000}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-17340}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-7200}
export HCCL_ASYNC_ERROR_HANDLING=${HCCL_ASYNC_ERROR_HANDLING:-0}
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-256}
export P2P_HCCL_BUFFSIZE=${P2P_HCCL_BUFFSIZE:-20}
export KIMI_RUN_TOKEN=${KIMI_RUN_TOKEN:-$(cat /proc/sys/kernel/random/uuid)}

export NPUS_PER_NODE=$NPUS_PER_NODE

KIMI_RAY_ARGS=()
if [[ "$RUN_MODE" == single ]]; then
    kimi_single_ray_args
else
    : "${RAY_ADDRESS:?multinode mode must be started by ray_start_multi_nodes_fsdpturbo_separate_async.sh}"
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
log_file="$LOG_DIR/${EXPERIMENT_NAME}.log"
rollout_data_dir="${VERL_ROLLOUT_DATA_DIR:-${PROJECT_ROOT}/rollout_dump/${RUN_TIMESTAMP}}"
mkdir -p "$rollout_data_dir"


DATA=(
    data.train_files="$TRAIN_FILE"
    data.val_files="$VAL_FILE"
    data.image_key=images
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.max_prompt_length=$MAX_PROMPT_LENGTH"
    "data.max_response_length=$MAX_RESPONSE_LENGTH"
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=null
    data.truncation=error
    data.shuffle=False
    data.validation_shuffle=False
    data.trust_remote_code=True
)

MODEL=(
    "actor_rollout_ref.model.path=$model_path"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=False
)

ACTOR=(
    actor_rollout_ref.actor.checkpoint.save_contents="['model','optimizer','extra']"
    actor_rollout_ref.actor.checkpoint.load_contents="['model','optimizer','extra']"
    "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-1e-6}"
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.optim.optimizer=AdamW
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$ppo_max_token_len"
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
)

# ── FSDP-Turbo config (actor + ref) ─────────────────────────────────────
fsdp_modules='{vision_tower.encoder.blocks.\{*\}:{},mm_projector:{},language_model.model.embed_tokens:{},language_model.model.layers.\{*\}.self_attn:{},language_model.model.layers.\{*\}.mlp:{},language_model.model.layers.\{*\}.block_sparse_moe:{},language_model.lm_head:{}}'
FSDP_CONFIG=()
for role in actor ref; do
    prefix="actor_rollout_ref.$role"
    FSDP_CONFIG+=(
        "$prefix.strategy=fsdp_turbo"
        "$prefix.fsdp_config.model_dtype=bfloat16"
        "$prefix.fsdp_config.dtype=bfloat16"
        "+$prefix.fsdp_config.turbo_config.model.attn_implementation=eager"
        "+$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.ignored_modules=[]"
        "++$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.apply_modules=$fsdp_modules"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.reduce_dtype=bf16"
        "$prefix.fsdp_config.turbo_config.distributed.fsdp_plan.fsdp_implementation=native"
        "$prefix.fsdp_config.turbo_config.distributed.fully_shard_parallel_size=$turbo_fsdp_size"
        "$prefix.fsdp_config.turbo_config.distributed.ulysses_parallel_size=1"
        "$prefix.fsdp_config.turbo_config.distributed.expert_parallel_size=$TURBO_EP_SIZE"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.apply_modules=['language_model.model.layers.{*}.block_sparse_moe.experts']"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.dispatcher=custom_ep_forward_fused"
        "$prefix.fsdp_config.turbo_config.distributed.expert_fully_shard_parallel_size=$turbo_efsdp_size"
        "+$prefix.fsdp_config.turbo_config.distributed.ep_plan.apply_efsdp_modules=['language_model.model.layers.{*}.block_sparse_moe.experts']"
        "$prefix.fsdp_config.reshard_after_forward=True"
        "$prefix.fsdp_config.offload_policy=False"
        "$prefix.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD"
        "$prefix.entropy_from_logits_with_chunking=True"
        "$prefix.use_torch_compile=False"
    )
done
FSDP_CONFIG+=(
    "actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD"
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_forward_prefetch=1
    actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.fsdp_plan.num_to_backward_prefetch=1
    +actor_rollout_ref.actor.fsdp_config.turbo_config.distributed.ep_plan.fixed_router=False
    +actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute=True
    '+actor_rollout_ref.actor.fsdp_config.turbo_config.memory.recompute_plan=["language_model.model.layers.{*}","vision_tower.encoder.blocks.{*}"]'
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0
    "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$ppo_max_token_len"
)

# ── Rollout config ──────────────────────────────────────────────────────
ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    "actor_rollout_ref.rollout.tensor_model_parallel_size=$rollout_tp_size"
    "actor_rollout_ref.rollout.data_parallel_size=$ROLLOUT_DP"
    "actor_rollout_ref.rollout.expert_parallel_size=$rollout_ep_size"
    "actor_rollout_ref.rollout.n=$ROLLOUT_N"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    "actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LENGTH"
    "actor_rollout_ref.rollout.max_num_seqs=$rollout_max_num_seqs"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$rollout_max_num_batched_tokens"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    "actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER"
    actor_rollout_ref.rollout.free_cache_engine=True
    "actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT:-dummy}"
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.logprobs_mode=processed_logprobs
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES:-256}"
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode=data
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_training_backend=fsdpturbo
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_training_causal_conv1d=True
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.kimi_kda_oproj_fp32_reduce=True
)

# Graph mode config (only when not eager)
case "${ROLLOUT_ENFORCE_EAGER,,}" in
    false|0|no|off)
        ROLLOUT+=(
            +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="$ROLLOUT_CUDAGRAPH_MODE"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_capture_sizes=$ROLLOUT_CUDAGRAPH_CAPTURE_SIZES"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.max_cudagraph_capture_size=$ROLLOUT_MAX_CUDAGRAPH_CAPTURE_SIZE"
            "+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.enable_sleep_mode_extra_cleanup=$ROLLOUT_SLEEP_EXTRA_CLEANUP"
        )
        ;;
esac

# ── Trainer config ──────────────────────────────────────────────────────
TRAINER=(
    trainer.critic_warmup=0
    'trainer.logger=["console"]'
    trainer.project_name="$PROJECT_NAME"
    trainer.experiment_name="$EXPERIMENT_NAME"
    "trainer.nnodes=$TRAINER_NNODES"
    "trainer.n_gpus_per_node=$SPLIT_TRAIN_GPUS"
    trainer.balance_batch=True
    trainer.val_before_train=False
    "trainer.resume_mode=$resume_mode"
    "trainer.save_freq=$save_freq"
    "trainer.max_actor_ckpt_to_keep=$max_actor_ckpt_to_keep"
    "trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP:-1}"
    "trainer.test_freq=${TEST_FREQ:--1}"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.total_training_steps=$total_training_steps"
    trainer.default_local_dir="$OUTPUT_ROOT"
    trainer.rollout_data_dir="$rollout_data_dir"
)

# ── Separate async specific config ──────────────────────────────────────
SEPARATE_ASYNC=(
    trainer.use_v1=True
    trainer.v1.trainer_mode=separate_async
    "trainer.v1.separate_async.parameter_sync_step=$PARAMETER_SYNC_STEP"
    "actor_rollout_ref.rollout.nnodes=$ROLLOUT_NNODES"
    "actor_rollout_ref.rollout.n_gpus_per_node=$SPLIT_ROLLOUT_GPUS"
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl
    # trainer.v1.separate_async.hybrid_rollout.enable_switch=False
    # actor_rollout_ref.hybrid_engine=False
)

TRANSFER_QUEUE=(
    "transfer_queue.backend.SimpleStorage.num_data_storage_units=$TQ_STORAGE_UNITS"
    transfer_queue.enable=True
)

# ── Ray env propagation ─────────────────────────────────────────────────
RAY=()
ray_env_vars=(
    TMPDIR XDG_CACHE_HOME HF_HOME HF_MODULES_CACHE TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR PYTHONDONTWRITEBYTECODE
    ASCEND_RT_VISIBLE_DEVICES PYTHONPATH OMP_NUM_THREADS PYTHONHASHSEED
    TOKENIZERS_PARALLELISM TQ_SIMPLE_STORAGE_SEND_RECV_TIMEOUT TQ_NUM_THREADS
    NON_MEGATRON FSDP_TURBO_ROOT VLLM_ASCEND_PATH
    VLLM_USE_V1 VLLM_VERSION VLLM_ASCEND_ENABLE_FLASHCOMM
    VLLM_ASCEND_ENABLE_NZ VLLM_DISABLE_COMPILE_CACHE
    VLLM_ALLOW_RUNTIME_LORA_UPDATING VLLM_BATCH_INVARIANT
    VLLM_LOGGING_LEVEL VERL_LOGGING_LEVEL VERL_KIMI_MLP_CHUNK_TOKENS
    CUDA_DEVICE_MAX_CONNECTIONS KIMI_RUN_TOKEN
    VLLM_ASCEND_TASK_QUEUE_ENABLE TASK_QUEUE_ENABLE
    HCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME HCCL_HOST_SOCKET_PORT_RANGE
    HCCL_NPU_SOCKET_PORT_RANGE HCCL_CONNECT_TIMEOUT HCCL_EXEC_TIMEOUT
    HCCL_ASYNC_ERROR_HANDLING HCCL_BUFFSIZE P2P_HCCL_BUFFSIZE
    HCCL_IF_BASE_PORT HCCL_OP_EXPANSION_MODE
    CPU_AFFINITY_CONF MULTI_STREAM_MEMORY_REUSE
)
for name in "${ray_env_vars[@]}"; do
    [[ -v "$name" ]] || continue
    RAY+=("++ray_kwargs.ray_init.runtime_env.env_vars.${name}=\"${!name}\"")
done

pipeline_summary=""
echo "run=$RUN_MODE model=$model_path train_world=$turbo_fsdp_size rollout_gpus=$SPLIT_ROLLOUT_GPUS FSDP=$turbo_fsdp_size EP=$TURBO_EP_SIZE eFSDP=$turbo_efsdp_size rollout_TP=$rollout_tp_size rollout_DP=$ROLLOUT_DP rollout_EP=$rollout_ep_size steps=$total_training_steps"
echo "separate_async parameter_sync_step=$PARAMETER_SYNC_STEP checkpoint_engine=nccl"
echo "actor_param_offload=$ACTOR_PARAM_OFFLOAD actor_optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD enforce_eager=$ROLLOUT_ENFORCE_EAGER rollout_gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION"
echo "micro_batch_actor=$ACTOR_MICRO_BATCH_SIZE_PER_GPU micro_batch_log_prob=$LOG_PROB_MICRO_BATCH_SIZE_PER_GPU"
echo "Kimi batches: train=$TRAIN_BATCH_SIZE rollout_n=$ROLLOUT_N effective=$((TRAIN_BATCH_SIZE * ROLLOUT_N)) ppo_mini=$PPO_MINI_BATCH_SIZE"
echo "Kimi sequence: prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH multimodal_margin=$rollout_multimodal_token_margin model_len=$MAX_MODEL_LENGTH actor_ref_tokens_per_gpu=$ppo_max_token_len"
echo "log=$log_file output=$OUTPUT_ROOT"

cd "$verl_path"
python3 -m verl.trainer.main_ppo \
    --config-path=config \
    --config-name=ppo_trainer.yaml \
    model_engine=dp \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${FSDP_CONFIG[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${SEPARATE_ASYNC[@]}" \
    "${TRANSFER_QUEUE[@]}" \
    "${RAY[@]}" \
    "${KIMI_RAY_ARGS[@]}" \
    "$@" 2>&1 | tee "$log_file"
