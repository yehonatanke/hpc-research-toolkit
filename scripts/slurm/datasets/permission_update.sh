#!/bin/bash

#SBATCH --job-name=permission_update_8k_9k
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/to_work/%x_%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/to_work/%x_%j.err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:30g

### BEFORE RUNNING THIS SCRIPT (or run the after 'all script') ###
# JOB_IDS=$(squeue -u $USER -h -o %i | tr '\n' ',' | sed 's/,$//')
# echo $JOB_IDS
# sbatch --dependency=afterany:$JOB_IDS $CODE/scripts/slurm/datasets/permission_update.sh

### CONFIGURATION ###
TARGET_DIR="$WORK/data/dl3dv_960"

echo "### STARTING PERMISSION UPDATE ON: $TARGET_DIR ###"

chmod -R g+rwX "$TARGET_DIR"

setfacl -R -m g::rwX "$TARGET_DIR"

setfacl -R -d -m g::rwx "$TARGET_DIR"

echo "### PERMISSIONS UPDATED SUCCESSFULLY ###"
getfacl "$TARGET_DIR"
echo "----------------------------------------"