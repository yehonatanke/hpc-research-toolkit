#!/bin/bash

# This script is used to implement `data_processing/viz_data.py` on a folder. 

# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=viz_data_folder
#SBATCH --output=${PROJECT_ROOT}/debug/logs/viz_data_folder/%j.out
#SBATCH --error=${PROJECT_ROOT}/debug/logs/viz_data_folder/%j.err
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=${ACCOUNT}

ROOT_DIR="${PROJECT_ROOT}"
DATASET_PATH="${ROOT_DIR}/debug/dl3dv_wai_dummy"
DATASET="dl3dv"
OUTPUT_DIR="${ROOT_DIR}/debug/output/viz_data_output"
SAVE_FILE="${OUTPUT_DIR}/test_1.rrd"
mkdir -p "$OUTPUT_DIR"

VENV="${ROOT_DIR}/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="${ROOT_DIR}/repos/map-anything"
LOG_DIR="${ROOT_DIR}/debug/logs/viz_data_folder"
mkdir -p "$LOG_DIR"

SECONDS=0

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "DATASET_PATH: $DATASET_PATH"
echo -e "DATASET: $DATASET"
echo -e "VENV: $VENV"
echo -e "MAPANYTHING_DIR: $MAPANYTHING_DIR"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[SLURM][INFO] Starting viz_data...\n"

cd "$MAPANYTHING_DIR"

echo "[SLURM][DEBUG] Current Directory: $(pwd)"

echo "[SLURM][INFO] Running script..."
for SCENE in "$DATASET_PATH"/*/; do
    if [ -d "$SCENE" ]; then
        echo -e "[SLURM][INFO] Processing scene: $SCENE"
        python3 data_processing/viz_data.py \
            --root_dir $DATASET_PATH \
            --scene $SCENE \
            --dataset $DATASET \
            --viz \
            --headless \
            --save $SAVE_FILE \
            --connect False \
            --create_save_file True \
            --debug 1
    fi
done

if [ $? -eq 0 ]; then
    echo -e "\n[SLURM][INFO][STATUS:Success] Viz_data completed successfully."
else
    echo -e "\n[SLURM][INFO][STATUS:Failure] Viz_data failed."
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."

echo -e "[SLURM][INFO] --- END OF JOB ---"
