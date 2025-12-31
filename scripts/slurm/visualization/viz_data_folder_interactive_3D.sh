#!/bin/bash
# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=viz_data_folder_interactive_3D
#SBATCH --output=${PROJECT_ROOT}/debug/output/viz_data/viz_data_interactive_3D/%j/job.out
#SBATCH --error=${PROJECT_ROOT}/debug/output/viz_data/viz_data_interactive_3D/%j/job.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=${ACCOUNT}

DESCRIPTION="Implements `data_processing/viz_data.py` on a folder with the addition of interactive_3D."
SCRIPT_NAME="viz_data_folder_interactive_3D.py"

# directories & job specifics
JOB_MSG_NAME="viz_data with interactive_3D"
ROOT_DIR="${PROJECT_ROOT}"
DATASET_DIR="${ROOT_DIR}/debug/dl3dv_wai_dummy"
OUTPUT_DIR="${ROOT_DIR}/debug/output/viz_data/viz_data_interactive_3D"
SAVE_FILE="${OUTPUT_DIR}/something.rrd"
DATASET="dl3dv"

mkdir -p "$OUTPUT_DIR"

# environment 
VENV="${ROOT_DIR}/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="${ROOT_DIR}/repos/map-anything"

# log directory
LOG_DIR="${ROOT_DIR}/debug/logs/viz_data_folder"

mkdir -p "$LOG_DIR"

SECONDS=0

echo -e "\n--- JOB INFO ---"
echo -e "DESCRIPTION: $DESCRIPTION"
echo -e "JOB"
echo -e "\t SCRIPT_NAME: $SCRIPT_NAME"
echo -e "\t LOG_DIR: $LOG_DIR"
# echo -e "- - - - - - - - - - - -"
echo -e "SYSTEM"
echo -e "\t USER: $(whoami)"
echo -e "\t NODE: $(hostname)"
echo -e "\t PATH: $(pwd)"
echo -e "\t DATE: $(date)"
# echo -e "- - - - - - - - - - - -"
echo -e "VARIABLES"
echo -e "\t ROOT_DIR: $ROOT_DIR"
echo -e "\t DATASET_DIR: $DATASET_DIR"
echo -e "\t OUTPUT_DIR: $OUTPUT_DIR"
echo -e "\t SAVE_FILE: $SAVE_FILE"
echo -e "\t DATASET: $DATASET"
echo -e "ENVIRONMENT"
echo -e "\t VENV: $VENV"
echo -e "\t MAPANYTHING_DIR: $MAPANYTHING_DIR"
echo -e "--------------------------------------\n"

echo -e "[SLURM][INFO] Starting viz_data...\n"

cd "$MAPANYTHING_DIR"

echo "[SLURM][DEBUG] Current Directory: $(pwd)"

echo "[SLURM][INFO] Running script..."
for SCENE in "$DATASET_DIR"/*/; do
    if [ -d "$SCENE" ]; then
        echo -e "[SLURM][INFO] Processing scene: $SCENE"
        python3 data_processing/viz_data.py \
            --root_dir $DATASET_DIR \
            --scene $SCENE \
            --dataset $DATASET \
            --viz \
            --headless \
            --save $SAVE_FILE \
            --connect False \
            --create_save_file True \
            --debug 1 \
            --interactive_3d True
    fi
done

if [ $? -eq 0 ]; then
    echo -e "\n[SLURM][INFO][STATUS::Success] Viz_data completed successfully."
else
    echo -e "\n[SLURM][INFO][STATUS::Failure] Viz_data failed."
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."

echo -e "--- END OF JOB ---"
