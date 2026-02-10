#!/bin/bash

#SBATCH --job-name=benchmark_clone_continue
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
REPO_URL="huggingface.co/datasets/DL3DV/DL3DV-Benchmark"

mkdir -p $TARGET_DIR
source $HOME/.bashrc
module load git-lfs/3.1.2

# increase the number of downloads in parallel
# on a strong server, you can try 64, but 32 is guaranteed and stable
export GIT_LFS_SKIP_SMUDGE=0
git config --global lfs.concurrenttransfers 32

echo "--- [Slurm] ${TITLE} ---"
echo -e "DATE: $(date)"
echo "NODE: $(hostname)"
echo "PATH: $(pwd)"
echo "TARGET_DIR: $TARGET_DIR"
echo -e "CONFIGS: \n- GIT_LFS_SKIP_SMUDGE: $GIT_LFS_SKIP_SMUDGE\n- lfs.concurrenttransfers: $lfs.concurrenttransfers"
echo "-----------------------------"

SECONDS=0

if [ -d "$TARGET_DIR" ]; then
    echo "DIRECTORY EXISTS. RESUMING DOWNLOAD (HIGH CONCURRENCY)..."
    cd "$TARGET_DIR"
    # verify that the settings also apply to the specific repository
    git config lfs.concurrenttransfers 32
    git lfs pull
else
    echo "DIRECTORY NOT FOUND. CLONING..."
    # Clone the structure + files in parallel according to the global setting
    git clone "https://oauth2:$HF_TOKEN@$REPO_URL" "$TARGET_DIR"
fi

DURATION=$SECONDS
echo -e "DURATION: $(($DURATION / 60)) MINUTES."