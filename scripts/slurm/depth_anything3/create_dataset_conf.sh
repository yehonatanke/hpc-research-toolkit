#!/bin/bash

# create dummy dataset with different confidence thresholds

#SBATCH --job-name=create_da3_dummy_dataset_conf_thresh_10
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/da3/create_dummy_dataset/dif_confs/10/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/da3/create_dummy_dataset/dif_confs/10/%j.err.log
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

CONF=10
TASK_NAME="create_da3_dummy_dataset_conf_thresh_${CONF}"

ROOT_DIR="${DEBUG}"
DATASET_PATH="${DEBUG}/dl3dv_wai_dummy"
IMAGES_DIR="${DATASET_PATH}"

OUTPUT_DIR="${DEBUG}/da3_dummy_dense__conf_${CONF}"

IMAGES_DISTORTED="images_distorted"

VENV="${ENVS}/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${REPOS}/Depth-Anything-3"
LOG_DIR="${DEBUG}/logs/da3/create_dummy_dataset/dif_confs/${CONF}"
MODEL_DIR="${REPOS}/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
DESCRIPTION="Run Depth Anything 3 on image folder with different confidence thresholds [conf=${CONF}]"

# Parameters
NUM_MAX_POINTS=2000000
PROCESS_RES=480
PROCESS_RES_METHOD="upper_bound_resize"
REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
CONF_THRESH_PERCENTILE=${CONF}
ALIGN_TO_INPUT_EXT_SCALE=True

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

LOG_MSG="[SLURM][INFO]"
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

echo -e "$LOG_MSG Running DA3 on each scene in the dataset..."
for SCENE_DIR in $DATASET_PATH/*/; do
    SCENE_NAME=$(basename "$SCENE_DIR")
    mkdir -p $OUTPUT_DIR/$SCENE_NAME
    
    echo -e "$LOG_MSG Scene Directory: $SCENE_DIR"
    echo -e "$LOG_MSG Images Directory: $IMAGES_DIR/$SCENE_NAME/$IMAGES_DISTORTED"
    echo -e "$LOG_MSG Output Directory: $OUTPUT_DIR/$SCENE_NAME"
    echo -e "$LOG_MSG RUNNING COMMAND: da3 auto $IMAGES_DIR/$SCENE_NAME/$IMAGES_DISTORTED --export-dir $OUTPUT_DIR/$SCENE_NAME --model-dir $MODEL_DIR --process-res $PROCESS_RES --process-res-method $PROCESS_RES_METHOD --num-max-points $NUM_MAX_POINTS --ref-view-strategy $REF_VIEW_STRATEGY --export-format $EXPORT_FORMAT --conf-thresh-percentile $CONF_THRESH_PERCENTILE --align-to-input-ext-scale --use-ray-pose --auto-cleanup"
        
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
    echo -e "\n$LOG_MSG [SUCCESS] '$TASK_NAME' completed successfully."
else
    echo -e "\n$LOG_MSG [ERROR] '$TASK_NAME' failed."
    exit 1
fi

duration=$SECONDS
echo -e "$LOG_MSG[TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo -e "[INFO] --- END OF JOB ---"
