#!/bin/bash
#SBATCH --job-name=check_statistics[11K]
#SBATCH --output=logs/check_statistics/11K/%j.out
#SBATCH --error=logs/check_statistics/11K/%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

ROOT_DIR="/leonardo_work/AIFAC_S02_060/data/yk"
SUBSET="11K"

DATA_DIR="${ROOT_DIR}/datasets/DL3DV-10K_workspace/DL3DV-10K_raw/${SUBSET}"
OUTPUT_FILE="${ROOT_DIR}/debug/analysis/resolution_analysis_results/${SUBSET}.json"
LOG_DIR="${ROOT_DIR}/debug/scripts/slurm/logs/check_statistics/${SUBSET}"
PYTHON_SCRIPT="${ROOT_DIR}/debug/scripts/python/check_statistics_v2.py"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$OUTPUT_FILE")"

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
echo -e "SUBSET: $SUBSET"
echo -e "DATA_DIR: $DATA_DIR"
echo -e "OUTPUT_FILE: $OUTPUT_FILE"
echo -e "LOG_DIR: $LOG_DIR"
echo -e "PYTHON_SCRIPT: $PYTHON_SCRIPT"
echo -e "-----------------------------\n"

echo -e "Starting analysis of image resolutions...\n"

python $PYTHON_SCRIPT \
    --data_dir "$DATA_DIR" \
    --output_file "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo -e "\n[SUCCESS] [SUBSET:$SUBSET] Analysis of image resolutions completed successfully.\n"
else
    echo -e "\n[ERROR] [SUBSET:$SUBSET] Analysis of image resolutions failed.\n"
fi

echo -e "--- END OF JOB ---"