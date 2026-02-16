#!/bin/bash

#SBATCH --job-name=download
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/acid/%x/acid2_out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/acid/%x/acid2_err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060
#SBATCH --gres=tmpfs:500g

FILE_NAME_1="acid.zip"
URL_1="http://schadenfreude.csail.mit.edu:8000/acid.zip"

FILE_NAME_2="acid_test_only.zip"
URL_2="http://schadenfreude.csail.mit.edu:8000/acid_test_only.zip"


TARGET_DIR="${WORK}/data/acid"
mkdir -p $TARGET_DIR


echo "--- [Slurm] DATE: $(date) ---"
echo "NODE: $(hostname)"
echo "PATH: $(pwd)"
echo "TARGET_DIR: $TARGET_DIR"
echo "-----------------------------"

SECONDS=0
curl -L "$URL_2" -o "${TARGET_DIR}/${FILE_NAME_2}"

DURATION=$SECONDS
echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 

if [ $? -eq 0 ]; then
    echo "--- Download Successful ---"
    echo "Dataset is available at: $TARGET_DIR"
    du -sh "$TARGET_DIR"
else
    echo "--- Download Failed ---"
fi
