#!/bin/bash

#SBATCH --job-name=create_dense_dataset_for_dl3dv960_debug_scene_poses
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/dl3dv960_dense/debug/100_scene_poses/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/dl3dv960_dense/debug/100_scene_poses/%j.err.log
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

source ${SLURM_UTILS}/_setup_sbatch.sh

### PARAMETERS ###
MAX_SCENES=5
OUT_NAME="100_scene_poses"

### DA3 DEBUG FLAGS ###
export DA3_LOG_LEVEL=DEBUG
export DL3DV_USE_SCENE_POSES=1

### PARAMETERS ###
PROCESS_RES=100 
PROCESS_RES_METHOD="upper_bound_resize"
REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
ALIGN_TO_INPUT_EXT_SCALE=True
### LOG DIR ###
LOG_DIR="${CODE}/scripts/logs/dl3dv960_dense/debug/${OUT_NAME}"

### JOB PATHS ###
DATASET_PATH="${SCRATCH}/DL3DV960_slurm/1K"
MODEL_DIR="${REPOS}/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
OUTPUT_DIR="${DEBUG}/dl3dv960_dense_debug/${OUT_NAME}"

### ENV ###
VENV="${ENVS}/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${REPOS}/Depth-Anything-3"

### MESSAGES ###
PRE_MSG="[DEBUG:SCENE_POSES]"
LIMIT_MSG="[LIMIT: ${MAX_SCENES}]"
### DESCRIPTION ###
TASK_NAME="${PRE_MSG}${LIMIT_MSG} [DEBUG SCENE_POSES] [DATASET: DL3DV960_slurm/1K]"
DESCRIPTION="${PRE_MSG}${LIMIT_MSG} Run Depth Anything 3 on DL3DV960 dataset [PROCESS_RES: 100] [CREATE DENSE DATASET] [USE SCENE POSES]"


### SETUP ###
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

### PRINT JOB HEADER ###
GENERAL_VARS=(DESCRIPTION DATASET_PATH OUTPUT_DIR MODEL_DIR DEPTH_ANYTHING_DIR LOG_DIR VENV)
PARAM_VARS=(PROCESS_RES PROCESS_RES_METHOD REF_VIEW_STRATEGY EXPORT_FORMAT ALIGN_TO_INPUT_EXT_SCALE MAX_SCENES)
_print_job_header

### --- MAIN PROCESS --- ###
SECONDS=0
source "$VENV"
cd "$DEPTH_ANYTHING_DIR"
SCENE_COUNT=0
for SCENE_DIR in $DATASET_PATH/*/; do
    if [ $SCENE_COUNT -ge $MAX_SCENES ]; then
        echo -e "$LOG_MSG REACHED LIMIT OF $MAX_SCENES SCENES. STOPPING ITERATION."
        break
    fi

    SCENE_NAME=$(basename "$SCENE_DIR")
    mkdir -p "$OUTPUT_DIR/$SCENE_NAME"
    
    ARGS=(
        auto "$DATASET_PATH/$SCENE_NAME/images_4"
        --export-dir "${OUTPUT_DIR}/${SCENE_NAME}"
        --model-dir "$MODEL_DIR"
        --process-res "$PROCESS_RES"
        --process-res-method "$PROCESS_RES_METHOD"
        --ref-view-strategy "$REF_VIEW_STRATEGY"
        --export-format "$EXPORT_FORMAT"
        --align-to-input-ext-scale
        --use-ray-pose
        --auto-cleanup
    )

    echo -e "${LOG_MSG} SCENE NAME: $SCENE_NAME"
    echo -e "${LOG_MSG} RUNNING COMMAND:"
    echo -e "${LOG_MSG} da3 $(print_args ARGS) \n"

    da3 "${ARGS[@]}"
    EXIT_CODE=$?

    echo ""
    if [ $EXIT_CODE -ne 0 ]; then
        echo -e "${ERR_MSG}[ERROR:SCENE_NUM: $SCENE_COUNT] DA3 FAILED ON SCENE: $SCENE_NAME"
    fi
    echo -e "${ERR_MSG}[SCENE_NUM: $SCENE_COUNT] AFTER DA3: SCENE=$SCENE_NAME EXIT_CODE=$EXIT_CODE"
    SCENE_COUNT=$((SCENE_COUNT + 1))
done

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${SYS_MSG} '$TASK_NAME' COMPLETED SUCCESSFULLY."
else
    echo -e "${ERR_MSG} '$TASK_NAME' FAILED."
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
LOG_DIR: "$LOG_DIR"
EOF


DURATION=$SECONDS
print_duration
echo -e "$END_OF_JOB_MSG"
