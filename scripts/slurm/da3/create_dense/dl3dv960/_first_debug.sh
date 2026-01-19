#!/bin/bash

#SBATCH --job-name=create_dense_dataset_for_dl3dv960_debugV1
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/dl3dv960_dense/debug/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/dl3dv960_dense/debug/%j.err.log
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

MAX_SCENES=5


PRE_MSG="[DEBUG-V1]"
LIMIT_MSG="[LIMIT: ${MAX_SCENES}]"
# general
TASK_NAME="${PRE_MSG}${LIMIT_MSG} create_dense_dataset_for_dl3dv960 [DATASET: DL3DV960_slurm/1K] [with Scale Factor and Continous Confidence (outlier_mask.npy)]"
DESCRIPTION="${PRE_MSG}${LIMIT_MSG} Run Depth Anything 3 on DL3DV960 dataset [create dense dataset] [for overfitting, to see if the model converges]"

# job paths
DATASET_PATH="${SCRATCH}/DL3DV960_slurm/1K"
MODEL_DIR="${REPOS}/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
# OUTPUT_DIR="${WORK}/data/dl3dv960_dense"
OUTPUT_DIR="${DEBUG}/dl3dv960_dense"

# env
VENV="${ENVS}/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${REPOS}/Depth-Anything-3"

# logs
# LOG_DIR="${CODE}/scripts/logs/dl3dv960_dense/DL3DV960_slurm"
LOG_DIR="${CODE}/scripts/logs/dl3dv960_dense/debug"

# Parameters
# NUM_MAX_POINTS=2000000    # last use: before scale and conf
# CONF_THRESH_PERCENTILE=40 # last use: before scale and conf

PROCESS_RES=966 ## TODO: check if this is correct
PROCESS_RES_METHOD="upper_bound_resize"
REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
ALIGN_TO_INPUT_EXT_SCALE=True

# fixed
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
LOG_MSG="[SLURM][INFO]"
DURATION_MSG="$LOG_MSG[TIME] Duration:"
END_OF_JOB_MSG="${LOG_MSG} --- END OF JOB ---"

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
echo "$PARAM_HEADER\n"
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

echo -e "${LOG_MSG} Running DA3 on up to 50 scenes in the dataset...\n"

# LIMIT 5
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

    da3 "${ARGS[@]}"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "$LOG_MSG [ERROR] da3 failed on scene $SCENE_NAME"
    fi

    SCENE_COUNT=$((SCENE_COUNT + 1))
done

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "$LOG_MSG [SUCCESS] '$TASK_NAME' completed successfully."
else
    echo -e "$LOG_MSG [ERROR] '$TASK_NAME' failed."
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

if [ $? -eq 0 ]; then
    echo -e "\n$LOG_MSG [SUCCESS] '$TASK_NAME' completed successfully."
else
    echo -e "\n$LOG_MSG [ERROR] '$TASK_NAME' failed."
    exit 1
fi

DURATION=$SECONDS
echo -e "$DURATION_MSG $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds."
echo -e "$END_OF_JOB_MSG"
