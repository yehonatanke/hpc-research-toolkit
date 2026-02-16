#!/bin/bash
#SBATCH --job-name=chunks
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/benchmark/%x/second/%a_out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/benchmark/%x/second/%a_err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=8G
#SBATCH --gres=tmpfs:100g
#SBATCH --array=0-9

### CONFIGURATION ###
CHUNKS_DIR="${CODE}/scripts/slurm/datasets/dl3dv/benckmark/hashes/chunks"   
DOWNLOAD_SCRIPT="${CODE}/scripts/slurm/datasets/dl3dv/benckmark/download_improve.py"    
TARGET_DIR="${SCRATCH}/DL3DV-10K-BENCHMARK-CHUNKS"
CHUNK_FILE="${CHUNKS_DIR}/chunk_0${SLURM_ARRAY_TASK_ID}"

mkdir -p "$TARGET_DIR"

module load python/3.11.7 
source $ENVS/download_dl3dv_subset/bin/activate

echo "STARTING JOB [ $SLURM_ARRAY_TASK_ID ] PROCESSING [ $CHUNK_FILE ]"

while IFS= read -r HASH; do
    # Skip if empty
    [ -z "$HASH" ] && continue

    # Check if already exists in final destination
    if [ -d "$TARGET_DIR/$HASH" ]; then
        echo "[SKIP] $HASH already exists."
        continue
    fi

    # Create unique local tmp dir for this specific scene
    # Using local scratch ($SLURM_TMPDIR) is faster if available, else $SCRATCH
    TMP_WORK_DIR="$SCRATCH/tmp/${SLURM_JOB_ID}/${HASH}"
    echo -e "[INFO] \n\t- TMP_WORK_DIR: ${TMP_WORK_DIR} \n\t- SLURM_JOB_ID: ${SLURM_JOB_ID}"
    
    mkdir -p "$TMP_WORK_DIR"

    echo "[START] Downloading $HASH to $TMP_WORK_DIR"

    SECONDS=0
    # We point --odir to TMP. Script will create structure: TMP/HASH/...
    python "$DOWNLOAD_SCRIPT" \
        --odir "$TMP_WORK_DIR" \
        --subset hash \
        --hash "$HASH" \
        --only_level4 \
        --clean_cache

    EXIT_CODE=$?
    DURATION=$SECONDS

    echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS."
    # Atomic Move (Only if python exited with 0)
    if [ $EXIT_CODE -eq 0 ]; then
        # The script creates the directory named after the Hash inside odir
        if [ -d "$TMP_WORK_DIR/$HASH" ]; then
            echo "[SUCCESS] Moving $HASH to final directory"
            mv "$TMP_WORK_DIR/$HASH" "$TARGET_DIR/"
            rm -rf "$TMP_WORK_DIR"
            echo -e "[DOWNLOAD COMPLETED] - OUTPUT DIRECTORY: ${TARGET_DIR}/${HASH}"
        else
            echo "[ERROR] Python success but folder $HASH not found in $TMP_WORK_DIR"
            # Keep tmp dir for debugging
            # rm -rf "$TMP_WORK_DIR"
        fi
    else
        echo "[FAIL] Download failed for $HASH. Cleaning up."
        rm -rf "$TMP_WORK_DIR"
    fi

done < "$CHUNK_FILE"

echo -e "JOB $SLURM_ARRAY_TASK_ID FINISHED."

