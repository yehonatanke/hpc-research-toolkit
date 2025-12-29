#!/bin/bash
#SBATCH --job-name=env_test
#SBATCH --output=$WORK/data/yk/debug/logs/env_test/%j.out
#SBATCH --error=$WORK/data/yk/debug/logs/env_test/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

echo "Testing Default Slurm Variables ---"
echo "HOME: $HOME"
echo "USER: $USER"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

echo "Testing Inherited Environment Variables ---"
# This will show if $WORK_DIR was exported from your terminal session
echo "WORK_DIR from environment: ${WORK_DIR:-'NOT DEFINED'}"

echo "Testing Positional Arguments ---"
# Run with: sbatch script.sh $WORK_DIR
echo "Argument 1: ${1:-'NO ARGUMENT PASSED'}"

echo "Testing Variables after Manual Sourcing ---"
# Explicitly loading .bashrc
if [ -f "$HOME/.bashrc" ]; then
    source "$HOME/.bashrc"
    echo "WORK_DIR after sourcing .bashrc: ${WORK_DIR:-'STILL NOT DEFINED'}"
else
    echo ".bashrc not found"
fi

echo "Current Working Directory ---"
pwd