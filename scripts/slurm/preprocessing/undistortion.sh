#!/bin/bash
# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=mapanything_undistortion
#SBATCH --output=${PROJECT_ROOT}/debug/logs/undistortion/%j.out
#SBATCH --error=${PROJECT_ROOT}/debug/logs/undistortion/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=${ACCOUNT}

ROOT="${PROJECT_ROOT}/debug/dl3dv_wai_dummy"
VENV="${PROJECT_ROOT}/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="${PROJECT_ROOT}/repos/map-anything"
CONFIGS="${MAPANYTHING_DIR}/data_processing/wai_processing/configs/undistortion/default.yaml"
LOG_DIR="${PROJECT_ROOT}/debug/logs/undistortion"

mkdir -p "$LOG_DIR"

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT: $ROOT"
echo -e "CONFIGS: $CONFIGS"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[INFO] Starting conversion...\n"

# module purge

# echo -e "[INFO] Activating venv..."
# source "$VENV"
# echo -e "[SUCCESS] Venv activated"

cd "$MAPANYTHING_DIR"

echo "[DEBUG] Current Directory: $(pwd)"

echo "[INFO] Running script..."
python -m wai_processing.scripts.undistort \
          "$CONFIGS" \
          root="$ROOT"


if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] Undistortion completed successfully."
else
    echo -e "\n[INFO] [ERROR] Undistortion failed."
fi

echo -e "[INFO] --- END OF JOB ---"