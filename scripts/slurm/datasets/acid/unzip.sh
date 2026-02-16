#!/bin/bash

#SBATCH --job-name=unzip
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/acid/%x/test_only_out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/acid/%x/test_only_err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060
#SBATCH --gres=tmpfs:500g

FILE_NAME_1="acid"
FILE_NAME_2="acid_test_only"

ZIP="${WORK}/data/acid/${FILE_NAME_2}.zip"
DEST="${WORK}/data/acid/${FILE_NAME_2}"

mkdir -p "$DEST"

echo "--- [Slurm] DATE: $(date) ---"
echo "NODE: $(hostname)"
echo "PATH: $(pwd)"
echo "DEST: $DEST"
echo -e "ZIP: $ZIP"
echo "-----------------------------"

unzip -q "$ZIP" -d "$DEST"

if [ $? -eq 0 ]; then
    echo "Extraction successful."
else
    echo "Extraction failed." >&2
    exit 1
fi