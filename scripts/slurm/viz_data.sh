#!/bin/bash
#SBATCH --job-name=viz_data
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/viz_data/%j.out
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/debug/logs/viz_data/%j.err
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
SCENE="1K_0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3"
DATASET="dl3dv"
OUTPUT_DIR="$ROOT_DIR/debug/viz_data_output"
mkdir -p "$OUTPUT_DIR"

VENV="$ROOT_DIR/envs/map-anything-venv/bin/activate"
MAPANYTHING_DIR="$ROOT_DIR/repos/map-anything"
LOG_DIR="$ROOT_DIR/debug/logs/viz_data"

mkdir -p "$LOG_DIR"

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "DATASET_PATH: $DATASET_PATH"
echo -e "SCENE: $SCENE"
echo -e "DATASET: $DATASET"
echo -e "VENV: $VENV"
echo -e "MAPANYTHING_DIR: $MAPANYTHING_DIR"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "-----------------------------\n"

echo -e "[INFO] Starting viz_data...\n"

# module purge

# echo -e "[INFO] Activating venv..."
# source "$VENV"
# echo -e "[SUCCESS] Venv activated"

cd "$MAPANYTHING_DIR"

echo "[DEBUG] Current Directory: $(pwd)"

echo "[INFO] Running script..."
python3 data_processing/viz_data.py \
    --root_dir $DATASET_PATH \
    --scene $SCENE \
    --dataset $DATASET \
    --viz \
    --depth_key pred_depth/mvsanywhere \
    --save /leonardo_work/AIFAC_S02_060/data/yk/debug/viz_data_output/test.rrd 
    # --save $OUTPUT_DIR/$SCENE.rrd

if [ $? -eq 0 ]; then
    echo -e "\n[INFO] [SUCCESS] Viz_data completed successfully."
else
    echo -e "\n[INFO] [ERROR] Viz_data failed."
fi

echo -e "[INFO] --- END OF JOB ---"