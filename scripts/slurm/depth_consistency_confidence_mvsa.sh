#!/bin/bash

# This script is used to implement `data_processing/scripts/depth_consistency_confidence.py` with MVSA.

#SBATCH --job-name=depth_consistency_confidence_mvsa
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/depth_consistency_confidence_mvsa/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/depth_consistency_confidence_mvsa/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

# specific
ROOT_DIR="/leonardo_work/AIFAC_S02_060/data/yk"
DATASET_PATH="$ROOT_DIR/debug/dl3dv_wai_dummy"
CONFIGS="$ROOT_DIR/repos/map-anything/data_processing/wai_processing/configs/depth_consistency_confidence/depth_consistency_confidence_mvsa.yaml" 

# general
VENV="$ROOT_DIR/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="$ROOT_DIR/repos/map-anything"
LOG_DIR="$ROOT_DIR/debug/logs/depth_consistency_confidence_mvsa"

mkdir -p "$LOG_DIR"

SECONDS=0

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "DATASET_PATH: $DATASET_PATH"
echo -e "CONFIGS: $CONFIGS"
echo -e "VENV: $VENV"
echo -e "MAPANYTHING_DIR: $MAPANYTHING_DIR"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[SLURM][INFO] Starting depth_consistency_confidence_mvsa...\n"

cd "$MAPANYTHING_DIR"

echo "[SLURM][DEBUG] Current Directory: $(pwd)"

echo -e "[SLURM][INFO] Running: python -m wai_processing.scripts.depth_consistency_confidence $CONFIGS root=$DATASET_PATH\n"
python -m wai_processing.scripts.depth_consistency_confidence \
          $CONFIGS \
          root=$DATASET_PATH

if [ $? -eq 0 ]; then
    echo -e "\n[SLURM][INFO][STATUS:Success] depth_consistency_confidence_mvsa completed successfully."
else
    echo -e "\n[SLURM][INFO][STATUS:Failure] depth_consistency_confidence_mvsa failed."
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."

echo -e "[SLURM][INFO] --- END OF JOB ---"
