#!/bin/bash
#SBATCH --job-name=change_resolution
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/change_resolution/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/change_resolution/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

ROOT="$WORK/data/yk/debug/dl3dv_wai_dummy"
LOG_DIR="$WORK/data/yk/debug/logs/change_resolution"
PYTHON_SCRIPT="$WORK/data/yk/debug/scripts/python/change_resolution.py"

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