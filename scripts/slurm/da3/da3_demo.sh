#!/bin/bash
#SBATCH --job-name=da3_demo
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/da3_demo/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/da3_demo/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

ROOT_DIR="/leonardo_work/AIFAC_S02_060/data/yk"
DATASET_PATH="$ROOT_DIR/debug/dl3dv_wai_dummy"
IMAGES_DIR="$DATASET_PATH/1K_0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images"
OUTPUT_DIR="$ROOT_DIR/debug/output/da3_demo_output"

VENV="$ROOT_DIR/envs/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="$ROOT_DIR/repos/Depth-Anything-3"
LOG_DIR="$ROOT_DIR/debug/logs/da3_demo"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# echo -e "[INFO] Starting Job on GPU Node: $(hostname)"
# echo -e "[INFO] Loading CUDA module..."
# module load cuda/12.1 

SECONDS=0

echo -e "[INFO] Activating venv..."
source "$VENV"

cd "$DEPTH_ANYTHING_DIR"
echo "[DEBUG] Current Directory: $(pwd)"



# Running MoGe 
echo -e "[SLURM][INFO] Running Depth Anything 3 Demo: \n\t- log dir: $LOG_DIR\n"

# da3 images $IMAGES_DIR --export-dir $OUTPUT_DIR --auto-cleanup

export MODEL_DIR=depth-anything/DA3NESTED-GIANT-LARGE
# This can be a Hugging Face repository or a local directory
# If you encounter network issues, consider using the following mirror: export HF_ENDPOINT=https://hf-mirror.com
# Alternatively, you can download the model directly from Hugging Face
export GALLERY_DIR=workspace/gallery
mkdir -p $GALLERY_DIR

# CLI auto mode with backend reuse
da3 backend --model-dir ${MODEL_DIR} --gallery-dir ${GALLERY_DIR} # Cache model to gpu
da3 auto assets/examples/SOH \
    --export-format glb \
    --export-dir ${GALLERY_DIR}/TEST_BACKEND/SOH \
    --use-backend


if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] Depth Anything 3 Demo completed successfully."
    echo -e "\n[INFO] [SUCCESS] gallery directory: $GALLERY_DIR"
else
    echo -e "\n[INFO] [ERROR] Depth Anything 3 Demo failed."
fi


duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo -e "[INFO] --- END OF JOB ---"