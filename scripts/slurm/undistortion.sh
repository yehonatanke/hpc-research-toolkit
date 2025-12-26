#!/bin/bash
#SBATCH --job-name=mapanything_undistortion
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/undistortion/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/undistortion/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

ROOT="/leonardo_work/AIFAC_S02_060/data/yk/debug/dl3dv_wai_dummy"
VENV="/leonardo_work/AIFAC_S02_060/data/yk/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="/leonardo_work/AIFAC_S02_060/data/yk/repos/map-anything"
CONFIGS="/leonardo_work/AIFAC_S02_060/data/yk/repos/map-anything/data_processing/wai_processing/configs/undistortion/default.yaml"
LOG_DIR="/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/undistortion"

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