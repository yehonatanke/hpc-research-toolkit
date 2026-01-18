#!/bin/bash

#SBATCH --job-name=Download_DL3DV960
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/download_dl3dv_960/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/download_dl3dv_960/%j.err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

# TARGET_DIR="${YK}/datasets/DL3DV960"

TARGET_DIR="${SCRATCH}/DL3DV960"
mkdir -p $TARGET_DIR

REPO_URL="huggingface.co/datasets/DL3DV/DL3DV-ALL-960P"

echo "--- [Slurm] DATE: $(date) ---"
echo "NODE: $(hostname)"
echo "PATH: $(pwd)"
echo "TARGET_DIR: $TARGET_DIR"
echo "-----------------------------"

source $HOME/.bashrc
module load git-lfs/3.1.2

git clone "https://oauth2:$HF_TOKEN@$REPO_URL" "$TARGET_DIR"

if [ $? -eq 0 ]; then
    echo "--- Download Successful ---"
    echo "Dataset is available at: $TARGET_DIR"
    du -sh "$TARGET_DIR"
else
    echo "--- Download Failed ---"
fi
