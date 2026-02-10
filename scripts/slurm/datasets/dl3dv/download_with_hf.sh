#!/bin/bash
#SBATCH --job-name=download_with_hf
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/%x/%j_out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/%x/%j_err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:500g
#SBATCH --account=AIFAC_S02_060


TITLE="DOWNLOADING THE FULL BENCHMARK WITH HF"

TARGET_DIR="${SCRATCH}/DL3DV-10K-BENCHMARK-HF"
mkdir -p $TARGET_DIR

echo "--- [Slurm] ${TITLE} ---"
echo -e "DATE: $(date)"
echo "NODE: $(hostname)"
echo "PATH: $(pwd)"
echo "TARGET_DIR: $TARGET_DIR"
echo "-----------------------------"

SECONDS=0

source "${ENVS}/hf_env/bin/activate"

# Enable fast transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
REPO_ID="DL3DV/DL3DV-Benchmark" 

cd "$TARGET_DIR"
hf download ${REPO_ID} \
    --repo-type dataset \
    --local-dir "$TARGET_DIR" \
    --token "$HF_TOKEN"

DURATION=$SECONDS
echo -e "DURATION: $(($DURATION / 60)) MINUTES AND $(($DURATION % 60)) SECONDS." 

if [ $? -eq 0 ]; then
    echo "--- DOWNLOAD COMPLETED ---"
    echo "- DATASET IS AVAILABLE AT: $TARGET_DIR"
    du -sh "$TARGET_DIR"
else
    echo "--- DOWNLOAD FAILED ---"
fi