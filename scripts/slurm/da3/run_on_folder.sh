#!/bin/bash
# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${YK:-${WORK:-$HOME}/project}}"

#SBATCH --job-name=create_da3_dummy_dataset
#SBATCH --output=${PROJECT_ROOT}/debug/logs/da3/create_dummy_dataset/%j.out.log
#SBATCH --error=${PROJECT_ROOT}/debug/logs/da3/create_dummy_dataset/%j.err.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=${ACCOUNT}


TASK_NAME="create_da3_dummy_dataset"

ROOT_DIR="${PROJECT_ROOT}"
DATASET_PATH="${ROOT_DIR}/debug/dl3dv_wai_dummy"
IMAGES_DIR="${DATASET_PATH}"

OUTPUT_DIR="${ROOT_DIR}/debug/da3_dummy/dense"

IMAGES_DISTORTED="images_distorted"

VENV="${ROOT_DIR}/envs/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${ROOT_DIR}/repos/Depth-Anything-3"
LOG_DIR="${ROOT_DIR}/debug/logs/da3/create_dummy_dataset"
MODEL_DIR="${ROOT_DIR}/repos/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
DESCRIPTION="Run Depth Anything 3 on image folder"

# Parameters
NUM_MAX_POINTS=2000000
PROCESS_RES=480
PROCESS_RES_METHOD="upper_bound_resize"
REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
CONF_THRESH_PERCENTILE=30
ALIGN_TO_INPUT_EXT_SCALE=True

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# echo -e "[INFO] Starting Job on GPU Node: $(hostname)"
# echo -e "[INFO] Loading CUDA module..."
# module load cuda/12.1 

SECONDS=0

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - - GENERAL- - - - - - - - - - - - "
echo -e "DESCRIPTION: $DESCRIPTION"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "DATASET_PATH: $DATASET_PATH"
echo -e "IMAGES_DIR: $IMAGES_DIR"
echo -e "OUTPUT_DIR: $OUTPUT_DIR"
echo -e "MODEL_DIR: $MODEL_DIR"
echo -e "DEPTH_ANYTHING_DIR: $DEPTH_ANYTHING_DIR"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "VENV: $VENV"
echo -e "- - - - - - - - - - - - PARAMETERS - - - - - - - - - - - -"
echo -e "NUM_MAX_POINTS: $NUM_MAX_POINTS"
echo -e "PROCESS_RES: $PROCESS_RES"
echo -e "PROCESS_RES_METHOD: $PROCESS_RES_METHOD"
echo -e "REF_VIEW_STRATEGY: $REF_VIEW_STRATEGY"
echo -e "EXPORT_FORMAT: $EXPORT_FORMAT"
echo -e "CONF_THRESH_PERCENTILE: $CONF_THRESH_PERCENTILE"
echo -e "ALIGN_TO_INPUT_EXT_SCALE: $ALIGN_TO_INPUT_EXT_SCALE"
echo -e "----------------------------------------------------------\n"

echo -e "[INFO] Activating venv..."
source "$VENV"

cd "$DEPTH_ANYTHING_DIR"
echo "[DEBUG] Current Directory: $(pwd)"

echo -e "[SLURM][INFO] Running DA3 on each scene in the dataset..."
for SCENE_DIR in $DATASET_PATH/*/; do
    SCENE_NAME=$(basename "$SCENE_DIR")
    mkdir -p $OUTPUT_DIR/$SCENE_NAME
    
    echo -e "[SLURM][INFO] Scene Directory: $SCENE_DIR"
    echo -e "[SLURM][INFO] Images Directory: $IMAGES_DIR/$SCENE_NAME/$IMAGES_DISTORTED"
    echo -e "[SLURM][INFO] Output Directory: $OUTPUT_DIR/$SCENE_NAME"
    echo -e "[SLURM][INFO] RUNNING COMMAND: da3 auto $IMAGES_DIR/$SCENE_NAME/$IMAGES_DISTORTED --export-dir $OUTPUT_DIR/$SCENE_NAME --model-dir $MODEL_DIR --process-res $PROCESS_RES --process-res-method $PROCESS_RES_METHOD --num-max-points $NUM_MAX_POINTS --ref-view-strategy $REF_VIEW_STRATEGY --export-format $EXPORT_FORMAT --conf-thresh-percentile $CONF_THRESH_PERCENTILE --align-to-input-ext-scale --use-ray-pose --auto-cleanup"
        
        da3 auto $IMAGES_DIR/$SCENE_NAME/$IMAGES_DISTORTED \
            --export-dir $OUTPUT_DIR/$SCENE_NAME \
            --model-dir $MODEL_DIR \
            --process-res $PROCESS_RES \
            --process-res-method $PROCESS_RES_METHOD \
            --num-max-points $NUM_MAX_POINTS \
            --ref-view-strategy $REF_VIEW_STRATEGY \
            --export-format $EXPORT_FORMAT \
            --conf-thresh-percentile $CONF_THRESH_PERCENTILE \
            --align-to-input-ext-scale \
            --use-ray-pose \
            --auto-cleanup

done

cd $OUTPUT_DIR
cat <<EOF > model_params.yaml
# PARAMETERS:
NUM_MAX_POINTS: $NUM_MAX_POINTS
PROCESS_RES: $PROCESS_RES
PROCESS_RES_METHOD: "$PROCESS_RES_METHOD"
REF_VIEW_STRATEGY: "$REF_VIEW_STRATEGY"
EXPORT_FORMAT: "$EXPORT_FORMAT"
CONF_THRESH_PERCENTILE: $CONF_THRESH_PERCENTILE
ALIGN_TO_INPUT_EXT_SCALE: ${ALIGN_TO_INPUT_EXT_SCALE,,}
EOF

if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] '$TASK_NAME' completed successfully."
else
    echo -e "\n[INFO] [ERROR] '$TASK_NAME' failed."
    exit 1
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo -e "[INFO] --- END OF JOB ---"


# Note: The following sections use environment variables $DA3_DUMMY and $DENSE_OVERLAY_OUT
# These should be set before running the script or replaced with ${PROJECT_ROOT}/... paths
# for SCENE_DIR in ${DA3_DUMMY:-${ROOT_DIR}/debug/da3_dummy}/*/; do
#     SCENE_NAME=$(basename "$SCENE_DIR") 
#     for FRAME in "${FRAMES[@]}"; do
#     echo -e ${SCENE_DIR}"rgb/dense/frame_${FRAME}.png" \
#     echo -e ${SCENE_DIR}"depth/dense/frame_${FRAME}.npy" \
#     echo -e ${DENSE_OVERLAY_OUT:-${ROOT_DIR}/debug/output}"DA3_DUMMY"/${SCENE_NAME} 
# done 
# done 

# # for da3 dummy (new)
# for SCENE_DIR in ${DA3_DUMMY:-${ROOT_DIR}/debug/da3_dummy}/*/; do
# SCENE_NAME=$(basename "$SCENE_DIR")
# for FRAME in "${FRAMES[@]}"; do
# run_depth_overlay \
# --rgb_path ${SCENE_DIR}"dense/rgb/frame_${FRAME}.png" \
# --depth_path ${SCENE_DIR}"dense/depth/frame_${FRAME}.npy" \
# --export-dir ${DENSE_OVERLAY_OUT:-${ROOT_DIR}/debug/output}"DA3_DUMMY"/${SCENE_NAME} \
# --model_name  depth-anything \
# --comment "CONF_THRESH_PERCENTILE=30, ALIGN_TO_INPUT_EXT_SCALE=True, PROCESS_RES=480, NUM_MAX_POINTS=2000000 PROCESS_RES_METHOD=upper_bound_resize" 
# done
# done