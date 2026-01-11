#!/bin/bash
#SBATCH --job-name=setup_dense_loader_venv
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/re10k_dense/setup_venv/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/re10k_dense/setup_venv/%j.err.log
#SBATCH --time=00:50:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

mkdir -p $CODE/scripts/logs/re10k_dense/setup_venv

INFO_TXT="[INFO]"
ERROR_TXT="[ERROR]"

if [ ! -d "$VENVS" ]; then
    echo -e "\n${ERROR_TXT} VENVS directory in: ${VENVS} not found."
    echo -e "\n${ERROR_TXT} Please create it and try again."
    exit 1
fi

ENV_NAME="dense_loader"
ENV_PATH="${VENVS}/${ENV_NAME}"

echo -e "${INFO_TXT} Environment '$ENV_NAME' is ready."

# Leonardo Specific Optimizations
export TORCH_CUDA_ARCH_LIST="8.0"

module purge
module load cuda/12.2
module load python/3.11.7
module load gcc/12.2.0

export FORCE_CUDA=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

PYTHON_BIN=$(which python3)

if [ ! -d "$ENV_PATH" ]; then
    echo -e "\n${INFO_TXT} Creating virtual environment: \n\t- at: ${ENV_PATH}\n\t- using: $PYTHON_BIN"
    $PYTHON_BIN -m venv "$ENV_PATH"
fi

source "$ENV_PATH/bin/activate"

echo -e "\n${INFO_TXT} Unsetting PYTHONPATH..."
unset PYTHONPATH
echo -e "\n${INFO_TXT} Upgrading pip..."

echo -e "\n${INFO_TXT} Reloading shell..."
hash -r


echo -e "\n${INFO_TXT} Environment '$ENV_NAME' is ready."

echo -e "\n--------------------------------------"
echo -e "${INFO_TXT} Environment '$ENV_NAME' is active."
echo -e "${INFO_TXT} Python: $(which python)"
echo -e "${INFO_TXT} Python Bin: $PYTHON_BIN"
echo -e "${INFO_TXT} Python Version: $(python --version)"
echo -e "${INFO_TXT} CUDA_HOME: $CUDA_HOME"
echo -e "--------------------------------------"

# Sequential Dependency Installation
# echo "Installing prerequisites..."
# pip install wheel setuptools
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

export MAX_JOBS=4

# echo "Building PyTorch3D from source (No Build Isolation)..."
# pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@75ebeeaea0908c5527e7b1e305fbc7681382db47"
echo -e "${INFO_TXT} Installing PyTorch3D from source (No Build Isolation)..."
cd $REPOS/_compute_node_downloads/pytorch3d
pip install --no-build-isolation -e .

echo -e "${INFO_TXT} Installing Flash Attention from source (No Build Isolation)..."
cd $REPOS/_compute_node_downloads/flash-attention
pip install --no-build-isolation -e .

# Install the rest of the requirements
REQ_FILE="${WORK}/code/vggt4d/data/re10k/requirements.txt"
if [ -f "$REQ_FILE" ]; then
    echo "Installing remaining requirements..."
    pip install --no-build-isolation -r "$REQ_FILE"
fi

echo "--- Setup Complete ---"
python -c 'import torch; import pytorch3d; print("PyTorch3D successfully installed.")'