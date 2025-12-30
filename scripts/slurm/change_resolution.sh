#!/bin/bash
# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=change_resolution
#SBATCH --output=${PROJECT_ROOT}/debug/logs/change_resolution/%j.out
#SBATCH --error=${PROJECT_ROOT}/debug/logs/change_resolution/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=${ACCOUNT}

ROOT="${PROJECT_ROOT}/debug/dl3dv_wai_dummy"
LOG_DIR="${PROJECT_ROOT}/debug/logs/change_resolution"
PYTHON_SCRIPT="${PROJECT_ROOT}/code/scripts/python/change_resolution.py"

mkdir -p "$LOG_DIR"

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT: $ROOT"
echo -e "PYTHON_SCRIPT: $PYTHON_SCRIPT"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[INFO] Starting change resolution...\n"

echo "[INFO] Running script..."
python "$PYTHON_SCRIPT" --root "$ROOT"

if [ $? -eq 0 ]; then
    echo -e "\n[SUCCESS] Change resolution completed successfully."
else
    echo -e "\n[ERROR] Change resolution failed."
fi

echo -e "[INFO] --- END OF JOB ---"