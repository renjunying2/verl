#!/usr/bin/env bash
# Multi-node Ray cluster launcher for Kimi-K3 FSDP-Turbo separate_async mode.
#
# Starts a Ray head + worker cluster across NNODES machines, then launches
# run_kimi_k3_fsdpturbo_separate_async.sh on the head node.
#
# Based on ray_start_multi_nodes_separate_async.sh, adapted for FSDP-Turbo backend.
#
# Usage:
#   # On head node (default 80.5.25.124):
#   bash ray_start_multi_nodes_fsdpturbo_separate_async.sh
#
#   # Specify nodes explicitly:
#   VERL_EXPECTED_NODE_IPS=80.5.25.124,80.5.25.128 bash ray_start_multi_nodes_fsdpturbo_separate_async.sh
#
#   # On worker nodes (run separately on each):
#   NODE_IP=80.5.25.128 VERL_EXPECTED_NODE_IPS=80.5.25.124,80.5.25.128 \
#       bash ray_start_multi_nodes_fsdpturbo_separate_async.sh
set -euo pipefail

kimi_runtime_init() {
    BACKEND_ROOT=${BACKEND_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd -P)}
    export BACKEND_ROOT
    # Training RPCs stay on the cluster network; proxies are only for dependency downloads.
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
    export PATH="$BACKEND_ROOT/.venv/bin:$BACKEND_ROOT/.tools/uv-aarch64-unknown-linux-gnu:$PATH"
    export PYTHONDONTWRITEBYTECODE=1
    export RAY_USAGE_STATS_ENABLED=0
    export HF_HUB_DISABLE_TELEMETRY=1
    export DO_NOT_TRACK=1
    # Use a node-local override for Triton/Ray temporary files when supplied.
    # The shared backend is NFS in this environment, so relying on the
    # repository's runtime-local directory still races during concurrent graph
    # compilation. The same pathname is local inside each container.
    if [[ -n "${KIMI_LOCAL_TMP_OVERRIDE:-}" ]]; then
        KIMI_LOCAL_TMP="$KIMI_LOCAL_TMP_OVERRIDE"
        KIMI_RAY_LOG_DIR="$KIMI_LOCAL_TMP/ray/$(hostname)/$$"
        export TMPDIR="$KIMI_LOCAL_TMP/tmp"
    else
        KIMI_RAY_LOG_DIR="$BACKEND_ROOT/t/$(hostname)/$$"
        KIMI_LOCAL_TMP="$BACKEND_ROOT/runtime-local"
        if mountpoint -q "$KIMI_LOCAL_TMP"; then
            export TMPDIR="$KIMI_LOCAL_TMP/tmp"
        else
            export TMPDIR="$KIMI_RAY_LOG_DIR"
        fi
    fi
    export XDG_CACHE_HOME="$BACKEND_ROOT/cache"
    export HF_HOME="$XDG_CACHE_HOME/huggingface"
    # Dynamic model packages are executable Python files. Shared cross-node writes
    # can leave importlib observing a package before its relative imports exist.
    # Triton metadata replacement on shared storage can also yield stale handles.
    # Keep executable caches on the same node-local mount in every container.
    export HF_MODULES_CACHE="$TMPDIR/huggingface_modules"
    export TRITON_CACHE_DIR="$TMPDIR/triton_cache"
    export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/torchinductor_cache"
    export UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
    export WANDB_DIR="$BACKEND_ROOT/run_artifacts"
    export KIMI_RUN_TOKEN=${KIMI_RUN_TOKEN:-kimi-$(date +%Y%m%d-%H%M%S)-$$}
    mkdir -p "$TMPDIR" "$KIMI_RAY_LOG_DIR" "$XDG_CACHE_HOME" "$WANDB_DIR"
    # Keep Ray's Unix sockets below Linux's 107-byte path limit. The directory
    # descriptor stays owned by this launcher and points inside BACKEND_ROOT.
    exec {KIMI_RAY_DIR_FD}< "$KIMI_RAY_LOG_DIR"
    export RAY_TMPDIR="/proc/$$/fd/$KIMI_RAY_DIR_FD"
    ulimit -c 0
}

kimi_single_ray_args() {
    # Ray belongs to the training driver's lifetime; no global ray stop needed.
    # Consumed by the backend launcher sourcing this helper.
    # shellcheck disable=SC2034
    KIMI_RAY_ARGS=(
        '++ray_kwargs.ray_init.address=local'
        "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR"
        "++ray_kwargs.ray_init.num_cpus=${RAY_NUM_CPUS:-64}"
        "++ray_kwargs.ray_init.resources.NPU=${NPUS_PER_NODE:-16}"
        "++ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_MEMORY:-8589934592}"
    )
}

# separate_async variant: starts Ray multi-node cluster and launches
# run_kimi_k3_fsdpturbo_separate_async.sh in multinode mode.
# Backend launchers source the shared setup; only direct execution starts Ray.
[[ "${BASH_SOURCE[0]}" == "$0" ]] || return 0

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
BACKEND_ROOT=${BACKEND_ROOT:-$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)}
KIMI_BACKEND=${KIMI_BACKEND:-fsdpturbo}
case "$KIMI_BACKEND" in fsdpturbo|megatron) ;; *) echo "KIMI_BACKEND must be fsdpturbo or megatron" >&2; exit 2 ;; esac
EXPECTED_NODE_IPS="${VERL_EXPECTED_NODE_IPS:-80.5.25.124,80.5.25.128,80.5.25.130,80.5.25.131}"
# EXPECTED_NODE_IPS="${VERL_EXPECTED_NODE_IPS:-80.5.25.124,80.5.25.128,80.5.25.130}"
IFS=, read -r -a expected_nodes <<< "$EXPECTED_NODE_IPS"
NNODES=${#expected_nodes[@]}
NPUS_PER_NODE=${NPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-${expected_nodes[0]}}
MASTER_PORT=${MASTER_PORT:-6781}
node_suffix="$(awk -F- '{print $NF}' /proc/sys/kernel/hostname)"
NODE_IP=${NODE_IP:-80.5.25.${node_suffix}}

if ((NNODES < 2)); then
    echo "VERL_EXPECTED_NODE_IPS must contain at least two nodes" >&2
    exit 2
fi
if [[ ",$EXPECTED_NODE_IPS," != *",$NODE_IP,"* ]]; then
    echo "node $NODE_IP is not in VERL_EXPECTED_NODE_IPS=$EXPECTED_NODE_IPS" >&2
    exit 2
fi

export NNODES NPUS_PER_NODE
export RAY_ADDRESS="$MASTER_ADDR:$MASTER_PORT"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NPUS_PER_NODE - 1)))}
token_nodes=${EXPECTED_NODE_IPS//,/--}
export KIMI_RUN_TOKEN=${KIMI_RUN_TOKEN:-kimi-${MASTER_ADDR}-${MASTER_PORT}-${token_nodes}}
kimi_runtime_init
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-103079215104}
# Tolerate transient raylet overload (shared/busy nodes) instead of letting
# the GCS falsely mark a healthy node as dead and drop it from the cluster.
export RAY_health_check_initial_delay_ms=${RAY_health_check_initial_delay_ms:-120000}
export RAY_health_check_period_ms=${RAY_health_check_period_ms:-10000}
export RAY_health_check_timeout_ms=${RAY_health_check_timeout_ms:-120000}
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-100000}
ulimit -n 65536

live_ray_processes="$(
    ps -C raylet,gcs_server -o stat=,pid=,args= 2>/dev/null \
        | awk '$1 !~ /^Z/ {print}' || true
)"
live_vllm_processes="$(
    ps -eo stat=,pid=,comm=,args= 2>/dev/null \
        | awk '$1 !~ /^Z/ && $3 ~ /^VLLM::/ {print}' || true
)"
# This is deliberately a *local* occupancy check.  On a late-joining worker,
# `ray status` can succeed by contacting an already-running remote head even
# though this node has no local Ray runtime; treating that as local occupancy
# prevents the missing worker from ever joining the cluster.
if [[ -n "$live_ray_processes" || -n "$live_vllm_processes" ]]; then
    echo "an existing Ray or VLLM runtime was found on $NODE_IP; refusing to replace it" >&2
    [[ -z "$live_ray_processes" ]] || echo "$live_ray_processes" >&2
    [[ -z "$live_vllm_processes" ]] || echo "$live_vllm_processes" >&2
    exit 3
fi

ray_owned=0
cleanup_owned_vllm() {
    local -a pids=()
    local environ_file pid comm
    for environ_file in /proc/[0-9]*/environ; do
        pid=${environ_file#/proc/}
        pid=${pid%/environ}
        [[ -r "$environ_file" && -r "/proc/$pid/comm" ]] || continue
        read -r comm < "/proc/$pid/comm" || continue
        [[ "$comm" == VLLM::* ]] || continue
        if tr '\0' '\n' < "$environ_file" 2>/dev/null \
            | grep -Fqx -- "KIMI_RUN_TOKEN=$KIMI_RUN_TOKEN"; then
            pids+=("$pid")
        fi
    done
    if ((${#pids[@]} == 0)); then
        return
    fi

    echo "Stopping ${#pids[@]} owned VLLM process(es) on $NODE_IP"
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 3
    for pid in "${pids[@]}"; do
        if [[ -r "/proc/$pid/environ" ]] && tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null \
            | grep -Fqx -- "KIMI_RUN_TOKEN=$KIMI_RUN_TOKEN"; then
            kill -KILL "$pid" 2>/dev/null || true
        fi
    done
}

cleanup_ray() {
    status=$?
    if [[ -n "${DIAG_PID:-}" ]]; then
        kill -TERM "$DIAG_PID" 2>/dev/null || true
        # Bound shutdown even if a GCS RPC is stuck.
        for _ in {1..10}; do
            kill -0 "$DIAG_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL "$DIAG_PID" 2>/dev/null || true
        wait "$DIAG_PID" 2>/dev/null || true
    fi
    if ((ray_owned == 1)); then
        echo "Stopping owned Ray runtime on $NODE_IP"
        python3 - "$KIMI_RUN_TOKEN" <<'PYCLEAN'
import os
import signal
import sys
import time
from pathlib import Path

token = ("KIMI_RUN_TOKEN=" + sys.argv[1]).encode()
skip = {os.getpid(), os.getppid()}
pids = []
for path in Path("/proc").iterdir():
    if not path.name.isdigit() or int(path.name) in skip:
        continue
    try:
        if token in (path / "environ").read_bytes().split(b"\0"):
            pids.append(int(path.name))
    except OSError:
        pass
for pid in pids:
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(2)
for pid in pids:
    try:
        if token in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
            os.kill(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
PYCLEAN
        cleanup_owned_vllm
    fi
    return "$status"
}
trap cleanup_ray EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ "$NODE_IP" == "$MASTER_ADDR" ]]; then
    ray start \
        --head \
        --temp-dir="$RAY_TMPDIR" \
        --port="$MASTER_PORT" \
        --node-ip-address="$NODE_IP" \
        --dashboard-host=0.0.0.0 \
        --dashboard-port="${RAY_DASHBOARD_PORT:-8260}" \
        --num-cpus="$RAY_NUM_CPUS" \
        --object-store-memory="$RAY_OBJECT_STORE_MEMORY" \
        --resources="{\"NPU\": $NPUS_PER_NODE}"
    ray_owned=1

    deadline=$((SECONDS + ${RAY_CLUSTER_START_TIMEOUT:-600}))
    until progress="$(python3 - "$MASTER_ADDR:$MASTER_PORT" "$EXPECTED_NODE_IPS" "$NPUS_PER_NODE" <<'PY'
import ray
import sys

address, expected_csv, npus_per_node = sys.argv[1:]
expected_ips = sorted(ip.strip() for ip in expected_csv.split(",") if ip.strip())
ray.init(address=address, logging_level="ERROR")
npu_count = int(ray.cluster_resources().get("NPU", 0))
available_npu = sum(
    int(info.get("NPU", 0))
    for info in ray._private.state.available_resources_per_node().values()
)
alive_ips = sorted(
    node["NodeManagerAddress"] for node in ray.nodes() if bool(node.get("Alive"))
)
ray.shutdown()
expected_npus = len(expected_ips) * int(npus_per_node)
print(
    f"Ray cluster: nodes={len(alive_ips)}/{len(expected_ips)} "
    f"NPU={npu_count}/{expected_npus} avail={available_npu}/{expected_npus} "
    f"connected={','.join(alive_ips) or 'none'}"
)
ready = (
    alive_ips == expected_ips
    and npu_count == expected_npus
    and available_npu == expected_npus
)
raise SystemExit(0 if ready else 1)
PY
    )"; do
        echo "$progress"
        ((SECONDS < deadline)) || { echo "Ray cluster startup timed out" >&2; exit 4; }
        sleep 5
    done
    echo "$progress"

    # Observe placement for the full training lifetime, not just startup.
    DIAG_LOG="$BACKEND_ROOT/verl/logs/ray_actor_distribution_$(date +%Y%m%d_%H%M%S)_${NODE_IP}_$$.jsonl"
    mkdir -p "$(dirname "$DIAG_LOG")"
    echo "Ray actor placement diagnostic log: $DIAG_LOG"
    python3 -u "$SCRIPT_DIR/diagnose_ray_placement.py" \
        --address "$MASTER_ADDR:$MASTER_PORT" --log "$DIAG_LOG" \
        --interval "${VERL_PLACEMENT_DIAG_INTERVAL:-30}" --runtime-probes &
    DIAG_PID=$!
    echo "Diagnostic sampler started (PID $DIAG_PID)"

    KIMI_RUN_MODE=multinode BACKEND_ROOT="$BACKEND_ROOT" \
        bash "$SCRIPT_DIR/run_kimi_k3_${KIMI_BACKEND}_separate_async.sh" "$@"
else
    echo "Ray worker $NODE_IP: connecting to $MASTER_ADDR:$MASTER_PORT"
    deadline=$((SECONDS + ${RAY_CLUSTER_START_TIMEOUT:-600}))
    # Keep the node-local socket path; Ray's CLI replaces a worker's --temp-dir.
    until python3 - "$MASTER_ADDR:$MASTER_PORT" "$NODE_IP" "$RAY_TMPDIR" \
        "$RAY_NUM_CPUS" "$NPUS_PER_NODE" "$RAY_OBJECT_STORE_MEMORY" <<'PYWORKER'
import sys

from ray._private.node import Node
from ray._private.parameter import RayParams

address, node_ip, temp_dir, num_cpus, num_npus, object_store_memory = sys.argv[1:]
params = RayParams(
    gcs_address=address,
    node_ip_address=node_ip,
    temp_dir=temp_dir,
    num_cpus=int(num_cpus),
    resources={"NPU": int(num_npus)},
    object_store_memory=int(object_store_memory),
)
node = Node(params, head=False, shutdown_at_exit=False, spawn_reaper=False)
node.check_version_info()
print(f"Ray worker started: {node_ip}, temp_dir={node.get_temp_dir_path()}", flush=True)
PYWORKER
    do
        ((SECONDS < deadline)) || { echo "Ray head startup timed out" >&2; exit 4; }
        echo "Ray worker $NODE_IP: waiting for head"
        sleep 5
    done
    ray_owned=1

    # `ray status` is a control-plane RPC, not a reliable liveness probe while
    # dozens of FSDP-Turbo workers are importing/building the model.  A short
    # GCS stall used to produce three false misses here and made this launcher
    # run `ray stop --force` on a healthy worker node.  Keep the launcher alive
    # for the lifetime of the local raylet instead; cluster loss is handled by
    # Ray itself, and the EXIT trap still cleans up only this launcher's runtime.
    raylet_pid="$(
        ps -C raylet -o stat=,pid= 2>/dev/null \
            | awk '$1 !~ /^Z/ {print $2; exit}'
    )"
    raylet_pid=${raylet_pid//[[:space:]]/}
    if [[ -z "$raylet_pid" ]]; then
        echo "Ray worker $NODE_IP: no live local raylet found after startup" >&2
        exit 4
    fi
    echo "Ray worker $NODE_IP: watching local raylet pid $raylet_pid"
    while [[ -r "/proc/$raylet_pid/stat" ]]; do
        read -r _ comm state _ < "/proc/$raylet_pid/stat" || break
        [[ "$comm" == "(raylet)" && "$state" != Z ]] || break
        sleep 10
    done
    echo "Ray worker $NODE_IP: local raylet exited"
fi
