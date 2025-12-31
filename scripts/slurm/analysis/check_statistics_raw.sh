#!/bin/bash
# Set PROJECT_ROOT to your project base directory (e.g., export PROJECT_ROOT=/path/to/project)
PROJECT_ROOT="${PROJECT_ROOT:-${WORK:-$HOME}/project}"

#SBATCH --job-name=check_statistics_raw
#SBATCH --output=logs/check_statistics_raw/%x_%j.out
#SBATCH --error=logs/check_statistics_raw/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=${ACCOUNT}

ROOT_DIR="${PROJECT_ROOT}"
DATA_DIR="${ROOT_DIR}/datasets/DL3DV-10K_workspace/DL3DV-10K_raw"
OUTPUT_FILE="${ROOT_DIR}/debug/analysis/resolution_analysis_results.json"
LOG_DIR="${ROOT_DIR}/code/scripts/slurm/logs/check_statistics_raw"
PYTHON_SCRIPT="${ROOT_DIR}/code/scripts/python/check_statistics.py"

mkdir -p "$LOG_DIR"

echo -e "\n--- [Slurm] DATE: $(date) ---"
echo -e "USER: $(whoami)"
echo -e "NODE: $(hostname)"
echo -e "PATH: $(pwd)"
echo -e "- - - - - - - - - - - -"
echo -e "ROOT_DIR: $ROOT_DIR"
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
    echo -e "\n[SUCCESS] Analysis of image resolutions completed successfully.\n"
else
    echo -e "\n[ERROR] Analysis of image resolutions failed.\n"
fi

echo -e "--- END OF JOB ---"