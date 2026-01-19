#!/bin/bash

#SBATCH --job-name=create_dense_dataset_for_dl3dv960_debug_518_skymask_roni
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/dl3dv960_dense/debug/100_skymask_softmax/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/dl3dv960_dense/debug/100_skymask_softmax/%j.err.log
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060


# treat unset variables and parameters as an error
set -u
# exit status is determined by the last command with a non-zero status
set -o pipefail

MAX_SCENES=5
OUT_NAME="518_roni_100"

PRE_MSG="[DEBUG-V1]"
LIMIT_MSG="[LIMIT: ${MAX_SCENES}]"
# general
TASK_NAME="${PRE_MSG}${LIMIT_MSG} create_dense_dataset_for_dl3dv960 [DATASET: DL3DV960_slurm/1K] [with Scale Factor and Continous Confidence (outlier_mask.npy) and Sky Mask (sky_mask.npz)]"
DESCRIPTION="${PRE_MSG}${LIMIT_MSG} Run Depth Anything 3 on DL3DV960 dataset [create dense dataset] [for overfitting, to see if the model converges]"

# job paths
DATASET_PATH="${SCRATCH}/DL3DV960_slurm/1K"
MODEL_DIR="${REPOS}/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
# OUTPUT_DIR="${WORK}/data/dl3dv960_dense"
OUTPUT_DIR="${DEBUG}/dl3dv960_dense_debug/${OUT_NAME}"

# env
VENV="${ENVS}/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${REPOS}/Depth-Anything-3"

# logs
# LOG_DIR="${CODE}/scripts/logs/dl3dv960_dense/DL3DV960_slurm"
LOG_DIR="${CODE}/scripts/logs/dl3dv960_dense/debug/${OUT_NAME}"

# Parameters
# NUM_MAX_POINTS=2000000    # last use: before scale and conf
# CONF_THRESH_PERCENTILE=40 # last use: before scale and conf

PROCESS_RES=100 ## TODO: check if this is correct 966
PROCESS_RES_METHOD="upper_bound_resize"
REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
ALIGN_TO_INPUT_EXT_SCALE=True

# fixed
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
LOG_MSG="[INFO]"
SYS_MSG="[DEBUG:SYS]"
GPU_MSG="[DEBUG:GPU]"
ERR_MSG="[DEBUG:ERROR]"
DURATION_MSG="[INFO:TIME] Duration:"
END_OF_JOB_MSG="${LOG_MSG} --- END OF JOB ---"

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

# main message
GENERAL_VARS=(DESCRIPTION DATASET_PATH OUTPUT_DIR MODEL_DIR DEPTH_ANYTHING_DIR LOG_DIR VENV)
PARAM_VARS=(PROCESS_RES PROCESS_RES_METHOD REF_VIEW_STRATEGY EXPORT_FORMAT ALIGN_TO_INPUT_EXT_SCALE MAX_SCENES)
GENERAL_HEADER="------------------------ GENERAL -------------------------"
PARAM_HEADER="------------------------- PARAMETERS -------------------------"
END_HEADER="----------------------------------------------------------"

cat <<EOF
--- [Slurm] DATE: $(date) ---
USER: $(whoami)
NODE: $(hostname)
PATH: $(pwd)
$GENERAL_HEADER
EOF
for var in "${GENERAL_VARS[@]}"; do
    printf "%s: %s\n" "$var" "${!var}"
done
echo "$PARAM_HEADER"
for var in "${PARAM_VARS[@]}"; do
    printf "%s: %s\n" "$var" "${!var}"
done
echo -e "$END_HEADER\n"

# --- jobs start - main process ---

SECONDS=0

echo -e "${LOG_MSG} Activating venv..."
source "$VENV"

cd "$DEPTH_ANYTHING_DIR"
echo -e "${LOG_MSG} Current Directory: $(pwd)"

print_sys_debug
print_gpu_debug

echo -e "\n${LOG_MSG} Running DA3 on ${MAX_SCENES} scenes in the dataset...\n"


SCENE_COUNT=0


for SCENE_DIR in $DATASET_PATH/*/; do
    if [ $SCENE_COUNT -ge $MAX_SCENES ]; then
        echo -e "$LOG_MSG Reached limit of $MAX_SCENES scenes. Stopping iteration."
        break
    fi

    SCENE_NAME=$(basename "$SCENE_DIR")
    mkdir -p "$OUTPUT_DIR/$SCENE_NAME"
    
    ARGS=(
        auto "$DATASET_PATH/$SCENE_NAME/images_4"
        --export-dir "$OUTPUT_DIR/$SCENE_NAME"
        --model-dir "$MODEL_DIR"
        --process-res "$PROCESS_RES"
        --process-res-method "$PROCESS_RES_METHOD"
        # --num-max-points "$NUM_MAX_POINTS"
        --ref-view-strategy "$REF_VIEW_STRATEGY"
        --export-format "$EXPORT_FORMAT"
        # --conf-thresh-percentile "$CONF_THRESH_PERCENTILE"
        --align-to-input-ext-scale
        --use-ray-pose
        --auto-cleanup
    )

    echo -e "$LOG_MSG Scene Name: $SCENE_NAME"
    echo -e "$LOG_MSG RUNNING COMMAND:"
    # printf " %s \\\n    " "${ARGS[@]}"
    echo "da3 $(print_args ARGS)"
    echo -e "\n"

    echo -e "${SYS_MSG}[SCENE_NUM: $SCENE_COUNT] Before da3: scene=$SCENE_NAME"
    print_gpu_debug

    # GPU_SAMPLER_LOG="${LOG_DIR}/${SLURM_JOB_ID:-nojob}_${SCENE_NAME}_gpu_sample.csv"
    # SEMPLE_TIME=1
    # GPU_SAMPLER_PID="$(start_gpu_sampler "$GPU_SAMPLER_LOG" "$SEMPLE_TIME")"
    # if [ -n "${GPU_SAMPLER_PID:-}" ]; then
    #     echo -e "${GPU_MSG} sampling nvidia-smi -> ${GPU_SAMPLER_LOG} (pid=${GPU_SAMPLER_PID})"
    # else
    #     echo -e "${GPU_MSG} sampler disabled (no nvidia-smi on PATH)"
    # fi

    da3 "${ARGS[@]}"
    
    EXIT_CODE=$?
    # stop_gpu_sampler "${GPU_SAMPLER_PID:-}"

    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "\n${ERR_MSG}[ERROR:SCENE_NUM: $SCENE_COUNT] da3 failed on scene $SCENE_NAME"
        echo -e "${ERR_MSG} After failure: scene=$SCENE_NAME exit_code=$EXIT_CODE"
        print_gpu_debug
        print_rss
    fi
    echo -e "${ERR_MSG}[SCENE_NUM: $SCENE_COUNT] After da3: scene=$SCENE_NAME exit_code=$EXIT_CODE"
    echo -e "${ERR_MSG} PRINT_GPU_DEBUG:"
    print_gpu_debug

    SCENE_COUNT=$((SCENE_COUNT + 1))
done

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${SYS_MSG} '$TASK_NAME' completed successfully."
else
    echo -e "${ERR_MSG} '$TASK_NAME' failed."
    exit $EXIT_CODE
fi

cd "$OUTPUT_DIR"
cat <<EOF > model_params.yaml
# DEPTH-ANYTHING-3 PARAMETERS:
NUM_MAX_POINTS: "NONE"
PROCESS_RES: $PROCESS_RES
PROCESS_RES_METHOD: "$PROCESS_RES_METHOD"
REF_VIEW_STRATEGY: "$REF_VIEW_STRATEGY"
EXPORT_FORMAT: "$EXPORT_FORMAT"
CONF_THRESH_PERCENTILE: "NONE"
ALIGN_TO_INPUT_EXT_SCALE: ${ALIGN_TO_INPUT_EXT_SCALE,,}
MAX_SCENES: $MAX_SCENES
EOF

# if [ $? -eq 0 ]; then
#     echo -e "\n$LOG_MSG [SUCCESS] '$TASK_NAME' completed successfully."
# else
#     echo -e "\n$LOG_MSG [ERROR] '$TASK_NAME' failed."
#     exit 1
# fi

DURATION=$SECONDS
echo -e "$DURATION_MSG $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds."
echo -e "$END_OF_JOB_MSG"
