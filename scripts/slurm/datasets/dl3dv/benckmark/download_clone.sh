#!/bin/bash

#SBATCH --job-name=benchmark_clone
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/download_dl3dv_benchmark/%x_%j/out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/download_dl3dv_benchmark/%x_%j/err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:500g
#SBATCH --account=AIFAC_S02_060

TITLE="DOWNLOADING THE FULL BENCHMARK WITH 960P RESOLUTION"

TARGET_DIR="${SCRATCH}/DL3DV-10K-BENCHMARK"
mkdir -p $TARGET_DIR

REPO_URL="huggingface.co/datasets/DL3DV/DL3DV-Benchmark"

echo "--- [Slurm] ${TITLE} ---"
echo -e "DATE: $(date)"
echo "NODE: $(hostname)"
echo "PATH: $(pwd)"
echo "TARGET_DIR: $TARGET_DIR"
echo "-----------------------------"

# lfs getstripe -d "${SCRATCH}/DL3DV-10K-BENCHMARK"
# lfs setstripe -c 1 "$TARGET_DIR"
source $HOME/.bashrc
module load git-lfs/3.1.2

SECONDS=0
git clone "https://oauth2:$HF_TOKEN@$REPO_URL" "$TARGET_DIR"
DURATION=$SECONDS

echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 

if [ $? -eq 0 ]; then
    echo "--- DOWNLOAD COMPLETED ---"
    echo "- DATASET IS AVAILABLE AT: $TARGET_DIR"
    du -sh "$TARGET_DIR"
else
    echo "--- DOWNLOAD FAILED ---"
fi
