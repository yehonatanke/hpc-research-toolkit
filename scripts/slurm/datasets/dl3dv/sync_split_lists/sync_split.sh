#!/bin/bash

#SBATCH --job-name=9k_01
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/sync/parts/%x_%j_out.log 
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/sync/parts/%x_%j_err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:100g

SUBSET=$1
PART=$2

SECONDS=0
SRC="${SCRATCH}/DL3DV960_DUP_EXT/${SUBSET}/"
DEST="${WORK}/data/dl3dv_960/${SUBSET}/"
LIST_DIR="${CODE}/scripts/slurm/datasets/dl3dv/sync_split_lists/${SUBSET}"
LIST_FILE="${LIST_DIR}/part_${PART}"

echo -e "STARTING AT [$(date)] - SIZE: $(du -sh "$DEST")"
echo -e "SUBSET: $SUBSET - PART: $PART"
echo -e "LIST FILE: $LIST_FILE"
echo -e "SOURCE: $SRC"
echo -e "DESTINATION: $DEST"
echo -e "--------------------------------"

rsync -rWh --stats --info=progress2 --files-from="$LIST_FILE" "$SRC" "$DEST" &
RSYNC_PID=$!

# Monitor progress every 5 minutes
while kill -0 $RSYNC_PID 2>/dev/null; do
    echo -e "CURRENT PROGRESS AT [$(date)] - SIZE: $(du -sh "$DEST")"
    sleep 300
done

# usage:
# sbatch $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 8K 00
# sbatch --dependency=afterany:33873708 $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 8K 01
# sbatch $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 9K 00
# sbatch --dependency=afterany:33874276 $CODE/scripts/slurm/datasets/dl3dv/sync_split_lists/sync_split.sh 9K 01

DURATION=$((SECONDS / 60))
echo -e "[SUCCESS] FINISHED AT [$(date)] - SIZE: $(du -sh "$DEST") - DURATION: $DURATION MINUTES"