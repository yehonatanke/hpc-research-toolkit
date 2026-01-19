#!/bin/bash

#SBATCH --job-name=create_dense_dataset_for_re10k_50_samples_dry
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/re10k_dense/50_samples/dry/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/re10k_dense/50_samples/dry/%j.err.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --mem=4G
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

# general
TASK_NAME="create_dense_dataset_for_re10k [50 samples] [dry run]"
DESCRIPTION="[dry run] Run Depth Anything 3 on RE10K dataset [create dense dataset] [50 samples] [for overfitting, to see if the model converges]"

# job paths
DATASET_PATH="${WORK}/data/re10k_precessed/test/test/images"
#IMAGES_DIR="${DATASET_PATH}"
OUTPUT_DIR="${WORK}/data/re10k_dense_da3_50_samples"
MODEL_DIR="${REPOS}/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"

# env
VENV="${ENVS}/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${REPOS}/Depth-Anything-3"

# logs
LOG_DIR="${CODE}/scripts/logs/re10k_dense/50_samples"

# Parameters
NUM_MAX_POINTS=2000000
PROCESS_RES=644
PROCESS_RES_METHOD="upper_bound_resize"
REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
CONF_THRESH_PERCENTILE=40
ALIGN_TO_INPUT_EXT_SCALE=True

# fixed
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
LOG_MSG="[SLURM][INFO]"
DRY_RUN_MSG="[DRY RUN]"
DURATION_MSG="$LOG_MSG[TIME] Duration:"
END_OF_JOB_MSG="${LOG_MSG} --- END OF JOB ---"

# main message
GENERAL_VARS=(DESCRIPTION DATASET_PATH OUTPUT_DIR MODEL_DIR DEPTH_ANYTHING_DIR LOG_DIR VENV)
PARAM_VARS=(NUM_MAX_POINTS PROCESS_RES PROCESS_RES_METHOD REF_VIEW_STRATEGY EXPORT_FORMAT CONF_THRESH_PERCENTILE ALIGN_TO_INPUT_EXT_SCALE)
GENERAL_HEADER="------------------------- GENERAL --------------------------"
PARAM_HEADER="------------------------- PARAMETERS -------------------------"
END_HEADER="----------------------------------------------------------"

cat <<EOF
--- [SLURM SESSION] DATE: $(date) ---
USER: $(whoami)
NODE: $(hostname)
PATH: $(pwd)
${GENERAL_HEADER}
EOF
for var in "${GENERAL_VARS[@]}"; do
    printf "%s: %s\n" "$var" "${!var}"
done
echo -e "${PARAM_HEADER}"
for var in "${PARAM_VARS[@]}"; do
    printf "%s: %s\n" "$var" "${!var}"
done
echo -e "${END_HEADER}\n"

# --- jobs start - main process ---

SECONDS=0

echo -e "${LOG_MSG} Activating venv..."
source "$VENV"

cd "$DEPTH_ANYTHING_DIR"
echo -e "${LOG_MSG} Current Directory: $(pwd)"

echo -e "${LOG_MSG} Running DA3 on each scene in the dataset..."

# ___ DRY RUN SECTION ___

# Define paths for testing
SCENE_DIRS=("$DATASET_PATH"/*/)
TOTAL_SCENES=${#SCENE_DIRS[@]}
LIMIT=50

echo -e "${DRY_RUN_MSG} SCENE_DIRS: $SCENE_DIRS"

echo -e "\nTotal scenes found: $TOTAL_SCENES"
echo "---------------------------------------"

# Dry run loop
ITER=1
for SCENE_DIR in "${SCENE_DIRS[@]:0:$LIMIT}"; do
    SCENE_NAME=$(basename "$SCENE_DIR")
    echo -e "${DRY_RUN_MSG} [$ITER/$LIMIT] READY TO PROCESS: $SCENE_NAME"
    ((ITER++))
done

echo "---------------------------------------"
echo -e "Dry run complete. Displayed $(($TOTAL_SCENES < $LIMIT ? $TOTAL_SCENES : $LIMIT)) scenes.\n"


echo -e "${LOG_MSG} Running DA3 on first $LIMIT scenes in the dataset..."
SCENE_DIRS=("$DATASET_PATH"/*/)
for SCENE_DIR in "${SCENE_DIRS[@]:0:$LIMIT}"; do
    SCENE_NAME=$(basename "$SCENE_DIR")
    # mkdir -p $OUTPUT_DIR/$SCENE_NAME
    echo -e "${DRY_RUN_MSG}\t - creating directory: $OUTPUT_DIR/$SCENE_NAME"
    
    ARGS=(
        auto "${DATASET_PATH}/${SCENE_NAME}"
        --export-dir "${OUTPUT_DIR}/${SCENE_NAME}"
        --model-dir "${MODEL_DIR}"
        --process-res "${PROCESS_RES}"
        --process-res-method "${PROCESS_RES_METHOD}"
        --num-max-points "${NUM_MAX_POINTS}"
        --ref-view-strategy "${REF_VIEW_STRATEGY}"
        --export-format "${EXPORT_FORMAT}"
        --conf-thresh-percentile "${CONF_THRESH_PERCENTILE}"
        --align-to-input-ext-scale
        --use-ray-pose
        --auto-cleanup
    )

    echo -e "${DRY_RUN_MSG} Scene Name: $SCENE_NAME"
    echo -e "${DRY_RUN_MSG} RUNNING COMMAND:"
    echo "da3 $(print_args ARGS)"
    echo -e "\n"

done

# ___ END OF DRY RUN SECTION ___

cd $OUTPUT_DIR
cat <<EOF > model_params.yaml
# ${DRY_RUN_MSG} DEPTH-ANYTHING-3 PARAMETERS:
NUM_MAX_POINTS: $NUM_MAX_POINTS
PROCESS_RES: $PROCESS_RES
PROCESS_RES_METHOD: "$PROCESS_RES_METHOD"
REF_VIEW_STRATEGY: "$REF_VIEW_STRATEGY"
EXPORT_FORMAT: "$EXPORT_FORMAT"
CONF_THRESH_PERCENTILE: $CONF_THRESH_PERCENTILE
ALIGN_TO_INPUT_EXT_SCALE: ${ALIGN_TO_INPUT_EXT_SCALE,,}
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
