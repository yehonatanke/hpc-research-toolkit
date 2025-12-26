#!/bin/bash
#SBATCH --job-name=run_moge_step1
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/run_moge/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/run_moge/%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

ROOT_DIR="/leonardo_work/AIFAC_S02_060/data/yk"
DATASET_PATH="$ROOT_DIR/debug/dl3dv_wai_dummy"
VENV="$ROOT_DIR/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="$ROOT_DIR/repos/map-anything"
LOG_DIR="$ROOT_DIR/debug/logs/run_moge"

mkdir -p "$LOG_DIR"

# echo -e "[INFO] Starting Job on GPU Node: $(hostname)"
# echo -e "[INFO] Loading CUDA module..."
# module load cuda/12.1 

SECONDS=0

echo -e "[INFO] Activating venv..."
source "$VENV"

cd "$MAPANYTHING_DIR"
echo "[DEBUG] Current Directory: $(pwd)"



# Running MoGe 
echo -e "[SLURM][INFO] Running MoGe: python -m wai_processing.scripts.run_moge root=$DATASET_PATH\n"

python -m wai_processing.scripts.run_moge \
          root=$DATASET_PATH \

if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] MoGe completed successfully."
else
    echo -e "\n[INFO] [ERROR] MoGe failed."
    exit 1
fi

duration=$SECONDS
echo -e "[SLURM][INFO][TIME] Duration: $(($duration / 60)) minutes and $(($duration % 60)) seconds."
echo -e "[INFO] --- END OF JOB ---"