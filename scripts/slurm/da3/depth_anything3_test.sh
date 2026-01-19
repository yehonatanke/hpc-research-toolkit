#!/bin/bash
# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=depth_anything3_test
#SBATCH --output=${PROJECT_ROOT}/debug/logs/depth_anything3_test/%j.out
#SBATCH --error=${PROJECT_ROOT}/debug/logs/depth_anything3_test/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=${ACCOUNT}

ROOT_DIR="${PROJECT_ROOT}"
DATASET_PATH="${ROOT_DIR}/debug/dl3dv_wai_dummy"
IMAGES_DIR="${DATASET_PATH}/1K_0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3/images"
OUTPUT_DIR="${ROOT_DIR}/debug/output/depth_anything3_test_output"

VENV="${ROOT_DIR}/envs/depth-anything-env/bin/activate"
DEPTH_ANYTHING_DIR="${ROOT_DIR}/repos/Depth-Anything-3"
LOG_DIR="${ROOT_DIR}/debug/logs/depth_anything3_test"

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
echo -e "[SLURM][INFO] Running Depth Anything 3 Test: da3 images $IMAGES_DIR --export-dir $OUTPUT_DIR\n"

da3 images $IMAGES_DIR --export-dir $OUTPUT_DIR --auto-cleanup

if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] Depth Anything 3 Test completed successfully."
else
    echo -e "\n[INFO] [ERROR] Depth Anything 3 Test failed."
    exit 1
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo -e "[INFO] --- END OF JOB ---"