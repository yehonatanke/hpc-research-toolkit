#!/bin/bash

# arg 1: environment name, arg 2: (optional) "no_color" 
ENV_NAME="$1"
NO_COLOR_MSG=0
if [[ "$2" == "no_color" || "$2" == "true" ]]; then
    NO_COLOR_MSG=1
fi

if [ "$NO_COLOR_MSG" -eq 1 ]; then
    BLUE_TEXT=""
    RED_TEXT=""
    NO_COLOR=""
else
    BLUE_TEXT="\033[34m"
    RED_TEXT="\033[31m"
    NO_COLOR="\033[0m"
fi

INFO_TXT="[${BLUE_TEXT}INFO${NO_COLOR}]"
ERROR_TXT="[${RED_TEXT}ERROR${NO_COLOR}]"

usage() {
    echo -e "${ERROR_TXT} Error: Environment name not provided."
    echo -e "${INFO_TXT} Usage: $0 <environment_name> [no_color|true]"
    exit 1
}

if [ $# -lt 1 ]; then
    usage
fi

if [ ! -d "$VENVS" ]; then
    echo -e "\n${ERROR_TXT} VENVS directory in: ${VENVS} not found."
    echo -e "\n${ERROR_TXT} Please create it and try again."
    exit 1
fi

ENV_PATH="${VENVS}/${ENV_NAME}"



echo -e "${INFO_TXT} Environment '$ENV_NAME' is ready."

# Leonardo Specific Optimizations
export TORCH_CUDA_ARCH_LIST="8.0"

module purge
module load cuda/12.2
module load python/3.11.7
module load gcc/12.2.0

# export CUDA_HOME=$CUDA_INSTALL_PATH
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
pip install --upgrade pip

echo -e "\n${INFO_TXT} Environment '$ENV_NAME' is ready."

echo -e "\n${BLUE_TEXT}--------------------------------------${NO_COLOR}"
echo -e "${INFO_TXT} Environment '$ENV_NAME' is active."
echo -e "${INFO_TXT} Python: $(which python)"
echo -e "${INFO_TXT} Python Bin: $PYTHON_BIN"
echo -e "${INFO_TXT} Python Version: $(python --version)"
echo -e "${INFO_TXT} CUDA_HOME: $CUDA_HOME"
echo -e "${BLUE_TEXT}--------------------------------------${NO_COLOR}"