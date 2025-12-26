#!/bin/bash
#SBATCH --job-name=da3_on_folder_[align_to_scale]_use_ray_pose_conf_70
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/da3/run_on_folder/[align_to_scale]_use_ray_pose_conf_70/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/da3/run_on_folder/[align_to_scale]_use_ray_pose_conf_70/%j.err
#SBATCH --time=00:35:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

TASK_NAME="da3_on_folder_[align_to_scale]_use_ray_pose_conf_70"
# TASK_NAME="da3_on_folder_[align_to_scale]_conf_40"

ROOT_DIR="/leonardo_work/AIFAC_S02_060/data/yk"
DATASET_PATH="$ROOT_DIR/debug/dl3dv_wai_dummy"
IMAGES_DIR="$DATASET_PATH/1K_0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images_distorted"
OUTPUT_DIR="$ROOT_DIR/debug/output/da3/dense/align_to_scale_use_ray_pose_conf_70"

VENV="$ROOT_DIR/envs/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="$ROOT_DIR/repos/Depth-Anything-3"
LOG_DIR="$ROOT_DIR/debug/logs/da3/run_on_folder/[align_to_scale]_use_ray_pose_conf_70"
MODEL_DIR="$ROOT_DIR/repos/Depth-Anything-3/models/DA3NESTED-GIANT-LARGE-1.1"
DESCRIPTION="Run Depth Anything 3 on image folder"

REF_VIEW_STRATEGY="saddle_sim_range"
EXPORT_FORMAT="dense"
CONF_THRESH_PERCENTILE=70
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
echo -e "REF_VIEW_STRATEGY: $REF_VIEW_STRATEGY"
echo -e "EXPORT_FORMAT: $EXPORT_FORMAT"
echo -e "CONF_THRESH_PERCENTILE: $CONF_THRESH_PERCENTILE"
echo -e "ALIGN_TO_INPUT_EXT_SCALE: $ALIGN_TO_INPUT_EXT_SCALE"
echo -e "----------------------------------------------------------\n"

echo -e "[INFO] Activating venv..."
source "$VENV"

cd "$DEPTH_ANYTHING_DIR"
echo "[DEBUG] Current Directory: $(pwd)"

# Running MoGe 
echo -e "[SLURM][INFO] RUNNINGCOMMAND: da3 auto $IMAGES_DIR --export-dir $OUTPUT_DIR --model-dir $MODEL_DIR --auto-cleanup --ref-view-strategy $REF_VIEW_STRATEGY --export-format $EXPORT_FORMAT --conf-thresh-percentile $CONF_THRESH_PERCENTILE\n"

da3 auto $IMAGES_DIR \
     --export-dir $OUTPUT_DIR \
     --model-dir $MODEL_DIR \
     --auto-cleanup \
     --ref-view-strategy $REF_VIEW_STRATEGY \
     --export-format $EXPORT_FORMAT \
     --conf-thresh-percentile $CONF_THRESH_PERCENTILE \
     --align-to-input-ext-scale \
     --use-ray-pose 


if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] '$TASK_NAME' completed successfully."
else
    echo -e "\n[INFO] [ERROR] '$TASK_NAME' failed."
    exit 1
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo -e "[INFO] --- END OF JOB ---"