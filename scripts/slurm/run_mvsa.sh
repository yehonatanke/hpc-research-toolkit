#!/bin/bash
#SBATCH --job-name=run_mvsa
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/run_mvsa/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/run_mvsa/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

ROOT_DIR="/leonardo_work/AIFAC_S02_060/data/yk"
DATASET_PATH="$ROOT_DIR/debug/dl3dv_wai_dummy"
VENV="$ROOT_DIR/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="$ROOT_DIR/repos/map-anything"
CONFIGS="$ROOT_DIR/repos/map-anything/data_processing/wai_processing/configs/covisibility/covisibility_pred_depth_mvsa.yaml" 
LOG_DIR="$ROOT_DIR/debug/logs/run_mvsa"

mkdir -p "$LOG_DIR"

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "DATASET_PATH: $DATASET_PATH"
echo -e "VENV: $VENV"
echo -e "MAPANYTHING_DIR: $MAPANYTHING_DIR"
echo -e "CONFIGS: $CONFIGS"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[INFO] Starting run_mvsa...\n"

module load cuda/12.8
# module purge

# echo -e "[INFO] Activating venv..."
# source "$VENV"
# echo -e "[SUCCESS] Venv activated"

cd "$MAPANYTHING_DIR"

echo "[DEBUG] Current Directory: $(pwd)"

echo "[INFO] Running script..."
python -m wai_processing.scripts.run_mvsanywhere \
          root="$DATASET_PATH"


if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] Run_mvsa completed successfully."
else
    echo -e "\n[INFO] [ERROR] Run_mvsa failed."
fi

echo -e "[INFO] --- END OF JOB ---"