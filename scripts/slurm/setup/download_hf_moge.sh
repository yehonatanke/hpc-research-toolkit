#!/bin/bash

# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=download_hf_moge
#SBATCH --output=${PROJECT_ROOT}/debug/logs/download_hf_moge/%j.out
#SBATCH --error=${PROJECT_ROOT}/debug/logs/download_hf_moge/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=${ACCOUNT}

# specific
ROOT_DIR="${PROJECT_ROOT}"
TARGET_DIR="${ROOT_DIR}/repos/map-anything/yk_moge"

mkdir -p "$TARGET_DIR"

# general
VENV="${ROOT_DIR}/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="${ROOT_DIR}/repos/map-anything"
LOG_DIR="${ROOT_DIR}/debug/logs/download_hf_moge"

mkdir -p "$LOG_DIR"

SECONDS=0

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "TARGET_DIR: $TARGET_DIR"
echo -e "VENV: $VENV"
echo -e "MAPANYTHING_DIR: $MAPANYTHING_DIR"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[SLURM][INFO] Starting download_hf_moge...\n"

cd "$MAPANYTHING_DIR"

echo "[SLURM][DEBUG] Current Directory: $(pwd)"

hf download Ruicheng/moge-2-vitl-normal --local-dir $TARGET_DIR

if [ $? -eq 0 ]; then
    echo -e "\n[SLURM][INFO][STATUS:Success] download_hf_moge completed successfully."
else
    echo -e "\n[SLURM][INFO][STATUS:Failure] download_hf_moge failed."
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."

echo -e "[SLURM][INFO] --- END OF JOB ---"
