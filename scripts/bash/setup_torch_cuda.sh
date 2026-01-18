#!/bin/bash

### SETUP TORCH CUDA FOR COMPILATION/TRAINING ###

# LOAD MODULES
module load cuda/12.1
module load gcc/12.2

# EXPORT CUDA ARCH LIST FOR NVIDIA A100
export TORCH_CUDA_ARCH_LIST="8.0"

TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent/'lib')")
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
export HDF5_USE_FILE_LOCKING=FALSE

### PRINT ENVIRONMENT CONFIGURATION ###
title_params=("=== ENVIRONMENT CONFIGURED ===" "\033[38;5;81m")
end_title_params=("=== END SETUP TORCH CUDA ===" "\033[38;5;81m")

echo ""
_title "${title_params[@]}"
echo -e "CUDA: 12.1"
echo -e "GCC: 12.2"
echo -e "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo -e "TORCH_LIB=$TORCH_LIB"
echo -e "HDF5_USE_FILE_LOCKING=FALSE"
_title "${end_title_params[@]}" 