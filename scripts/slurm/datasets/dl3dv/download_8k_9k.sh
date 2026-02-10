#!/bin/bash

#SBATCH --job-name=download_dl3dv_8k_9k
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/%x/%j/9k_out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/%x/%j/9k_err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

SUBSET="9K"
RESOLUTION="960P"
SCRIPT="$CODE/scripts/slurm/datasets/dl3dv/download.py" 
TARGET_DIR="${SCRATCH}/DL3DV-10K-CHECK-8K-9K"
mkdir -p $TARGET_DIR

source $ENVS/download_dl3dv_subset/bin/activate

SECONDS=0
python $SCRIPT --subset $SUBSET --resolution $RESOLUTION --file_type images+poses --odir $TARGET_DIR # --clean_cache
DURATION=$SECONDS

if [ $? -eq 0 ]; then
    echo -e "DOWNLOAD COMPLETED. \n\t- OUTPUT DIRECTORY: ${TARGET_DIR}"
    echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 
else
    echo -e "DOWNLOAD FAILED. \n\t- OUTPUT DIRECTORY: ${TARGET_DIR}"
    echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 
    exit 1
fi