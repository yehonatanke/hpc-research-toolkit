#!/bin/bash

#SBATCH --job-name=download_dl3dv_benchmark
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/%x/optimized/%j/out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/%x/optimized/%j/err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:1000g


TITLE="DOWNLOADING THE FULL BENCHMARK WITH OPTIMIZED DOWNLOAD"

### CONFIGURATION ###
SCRIPT="$CODE/scripts/slurm/datasets/dl3dv/download_benchmark_opt2.py" 
TARGET_DIR="${SCRATCH}/DL3DV-10K-BENCHMARK-OPT"
mkdir -p $TARGET_DIR

source $ENVS/download_dl3dv_subset/bin/activate

echo -e "### ${TITLE} ###"
echo -e "[$(date)] RUNNING COMMAND: python3 $SCRIPT --subset full --odir $TARGET_DIR --only_level4 --workers 128"

SECONDS=0
python3 $SCRIPT --subset full --odir $TARGET_DIR --only_level4 # --workers 8
DURATION=$SECONDS

if [ $? -eq 0 ]; then
    echo -e "DOWNLOAD COMPLETED. \n\t- OUTPUT DIRECTORY: ${TARGET_DIR}"
    echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 
else
    echo -e "DOWNLOAD FAILED. \n\t- OUTPUT DIRECTORY: ${TARGET_DIR}"
    echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 
    exit 1
fi