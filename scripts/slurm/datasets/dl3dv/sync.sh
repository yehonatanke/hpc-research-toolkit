#!/bin/bash

#SBATCH --job-name=sync_9k
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/%x/%j/out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/dl3dv/%x/%j/err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:400g

SUBSET="9K"
SRC="${SCRATCH}/DL3DV960_DUP_EXT"
DEST="${WORK}/data/dl3dv_960/"

rsync -rW --info=progress2 ${SRC}/${SUBSET} ${DEST}