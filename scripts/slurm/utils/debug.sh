# treat unset variables and parameters as an error
set -u
# exit status is determined by the last command with a non-zero status
set -o pipefail


print_sys_debug() {
    echo -e "\n===== SYSTEM DEBUG ====="
    echo "${SYS_MSG} hostname: $(hostname)"
    echo "${SYS_MSG} pwd: $(pwd)"
    echo "${SYS_MSG} SLURM_JOB_ID: SLURM_JOB_ID=${SLURM_JOB_ID:-unset} SLURM_STEP_ID=${SLURM_STEP_ID:-unset} SLURM_PROCID=${SLURM_PROCID:-unset}"
    echo "${SYS_MSG} CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
    echo "${SYS_MSG} PYTHON: $(command -v python || true)"
    echo "${SYS_MSG} DA3: $(command -v da3 || true)"
    echo "${SYS_MSG} ulimit:"
    ulimit -a || true
    echo "${SYS_MSG} free -h:"
    free -h || true
    echo -e "===== END OF SYSTEM DEBUG =====\n"
}

print_gpu_debug() {
    echo -e "\n===== GPU DEBUG ====="
    if command -v nvidia-smi >/dev/null 2>&1; then
        echo "${GPU_MSG} nvidia-smi (summary):"
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true
        echo "${GPU_MSG} nvidia-smi (processes):"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
    else
        echo "${GPU_MSG} nvidia-smi not found on PATH"
    fi

    # Optional: torch-level view (only works after venv activation and if CUDA is available)
    python - <<'PY' || true
import os
try:
    import torch
    print("[DEBUG:TORCH]", torch.__version__)
    print("[DEBUG:TORCH] cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        print(f"[DEBUG:TORCH] device {i}: {props.name}, total_mem_gb={props.total_memory/1024**3:.2f}")
        print("[DEBUG:TORCH] mem_allocated_gb:", torch.cuda.memory_allocated(i)/1024**3)
        print("[DEBUG:TORCH] mem_reserved_gb:", torch.cuda.memory_reserved(i)/1024**3)
except Exception as e:
    print("[DEBUG:TORCH] failed:", repr(e))
print("[DEBUG:ENV] CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("[DEBUG:ENV] PYTORCH_CUDA_ALLOC_CONF=", os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
print("[DEBUG:ENV] PYTORCH_ALLOC_CONF=", os.environ.get("PYTORCH_ALLOC_CONF"))
PY
    echo -e "===== END OF GPU DEBUG =====\n"
}

start_gpu_sampler() {
    # Usage: start_gpu_sampler <output_csv> [interval_seconds]
    # Writes periodic GPU memory snapshots to output_csv, returns PID via stdout.
    local out_csv="$1"
    local interval="${2:-1}"

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo ""
        return 0
    fi

    mkdir -p "$(dirname "$out_csv")"
    echo "unix_ts,gpu_index,mem_used_mib,mem_free_mib,util_gpu_pct,util_mem_pct" >> "$out_csv"

    (
        while true; do
            # One line per GPU; for --gres=gpu:1 this is usually a single line.
            # Note: timestamp from nvidia-smi is locale-dependent, so we log unix_ts ourselves.
            local ts
            ts="$(date +%s)"
            nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits \
              | awk -v ts="$ts" -F',' '{gsub(/ /,"",$0); print ts","$1","$2","$3","$4","$5}' \
              >> "$out_csv" 2>/dev/null
            sleep "$interval"
        done
    ) &

    echo $!
}

stop_gpu_sampler() {
    # Usage: stop_gpu_sampler <pid>
    local pid="${1:-}"
    if [ -z "$pid" ]; then
        return 0
    fi
    if kill -0 "$pid" >/dev/null 2>&1; then
        kill "$pid" >/dev/null 2>&1 || true
        wait "$pid" >/dev/null 2>&1 || true
    fi
}

print_rss() {
    # show the biggest RSS processes 
    echo -e "\n${ERR_MSG} top RSS processes:"
    # ps -eo pid,ppid,cmd,comm,rss --sort=-rss | head -n 20 || true
    # ps -eo pid,ppid,rss,comm --sort=-rss | head -n 20 | awk 'NR==1 {print $1, $2, "RSS(MiB)", $4} NR>1 {printf "%-7s %-7s %-10.2f %-s\n", $1, $2, $3/1024, $4}' | column -t
    ps -eo pid,ppid,rss,cmd --sort=-rss | head -n 20 | awk 'NR==1 {print $1, $2, "RSS(GiB)", $4} NR>1 {printf "%-7s %-7s %-10.2f %-s\n", $1, $2, $3/1024/1024, $4}' | column -t
}