#!/bin/bash

#SBATCH --job-name=2
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/download_benchmark/clone/continue/%x_out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/download_benchmark/clone/continue/%x_err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:1000g
#SBATCH --account=AIFAC_S02_060

# --- Configuration ---
TITLE="DOWNLOADING THE FULL BENCHMARK WITH 960P RESOLUTION"
# Ensure SCRATCH is defined, or use absolute path
TARGET_DIR="${SCRATCH}/DL3DV-10K-BENCHMARK" 
REPO_URL="huggingface.co/datasets/DL3DV/DL3DV-Benchmark"

# --- Setup ---
source $HOME/.bashrc
module load git-lfs/3.1.2 

# --- Dynamic Thread Allocation ---
# Check if SLURM_CPUS_PER_TASK is set and greater than 0
if [[ -n "$SLURM_CPUS_PER_TASK" ]] && [[ "$SLURM_CPUS_PER_TASK" -gt 0 ]]; then
    THREADS=$SLURM_CPUS_PER_TASK
    echo ">>> DETECTED SLURM CPU ALLOCATION. USING [ $THREADS ] PARALLEL THREADS."
else
    THREADS=8
    echo ">>> NO SLURM CPU ALLOCATION DETECTED. DEFAULTING TO [ $THREADS ] PARALLEL THREADS."
fi

# Apply Git LFS configurations
git config --global lfs.concurrenttransfers $THREADS
# Increase buffer to prevent hanging on large writes
git config --global http.postBuffer 524288000
# Timeout handling for stalled connections
git config --global lfs.activitytimeout 60

echo "--- [Slurm] ${TITLE} ---"
echo "DATE: $(date)"
echo "NODE: $(hostname)"
echo "TARGET_DIR: $TARGET_DIR"
echo "THREADS: $THREADS"
echo "-----------------------------"

SECONDS=0


# Check for .git directory to determine if repo is valid
if [ -d "$TARGET_DIR/.git" ]; then
    echo ">>> REPOSITORY DETECTED. RESUMING DOWNLOAD..."
    cd "$TARGET_DIR"
    
    # Ensure smudge is enabled for the pull phase
    export GIT_LFS_SKIP_SMUDGE=0

else
    echo ">>> REPOSITORY NOT FOUND. STARTING FRESH CLONE..."
    
    # Safety cleanup to ensure git clone doesn't fail on non-empty dir
    # rm -rf "$TARGET_DIR"
    
    # Clone only metadata/pointers (skips large files initially)
    # This prevents the initial clone from timing out
    export GIT_LFS_SKIP_SMUDGE=1
    
    git clone "https://oauth2:$HF_TOKEN@$REPO_URL" "$TARGET_DIR"
    
    if [ $? -ne 0 ]; then
        echo ">>> GIT CLONE FAILED. EXITING."
        exit 1
    fi
    
    cd "$TARGET_DIR"
    
    # Re-enable smudge for the actual content download
    export GIT_LFS_SKIP_SMUDGE=0
    
    # Install local hooks
    git lfs install --local
fi

# --- Execution ---

echo ">>> STARTING LFS PULL..."
git lfs pull

echo "-----------------------------"
DURATION=$SECONDS
echo "FINISHED. DURATION: $(($DURATION / 60)) MINUTES."