#!/bin/bash

#SBATCH --job-name=sync_benchmark
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/benchmark/sync_%j/out.log 
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/sync/parts/sync_%j/err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:1000g

SECONDS=0
SRC="${SCRATCH}/DL3DV-10K-BENCHMARK-CHUNKS"
DEST="${WORK}/data/dl3dv_banchmark_orig"

mkdir -p $DEST

echo -e "STARTING AT [$(date)] - SIZE: $(du -sh "$DEST")"
echo -e "SOURCE: $SRC"
echo -e "DESTINATION: $DEST"
echo -e "--------------------------------"

rsync -rWh --stats --info=progress2 "$SRC" "$DEST"

EXIT_CODE=$?
DURATION=$((SECONDS / 60))

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "--------------------------------"
    echo -e "SUCCESS: Transfer completed in $DURATION minutes."
    echo -e "FINAL SIZE: $(du -sh "$DEST")"
else
    echo -e "--------------------------------"
    echo -e "[DURATION: $DURATION MINUTES] FAILURE: rsync exited with code $EXIT_CODE" >&2
    exit $EXIT_CODE
fi

# usage:
# sbatch $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 8K 00
# sbatch --dependency=afterany:33873708 $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 8K 01
# sbatch $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 9K 00
# sbatch --dependency=afterany:33874276 $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 9K 01

