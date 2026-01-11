#!/bin/bash

#SBATCH --job-name=debug_loader_for_re10k_min_env
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/re10k_dense/debug_loader/min_env/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/re10k_dense/debug_loader/min_env/%j.err.log
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --account=AIFAC_S02_060

# general
TASK_NAME="debug loader for re10k min env"
DESCRIPTION="Debug loader for re10k min env"

# job paths
SCRIPT_PATH=$WORK/code/vggt4d/data/re10k
#  SCENE_PATH=$WORK/data/re10k_dense_da3_50_samples/00a5a2af678f37d5
SCENE_PATH=$WORK/data/re10k_dense_da3_50_samples

# env
# VENV=${WORK}/envs/4DCLF/bin/activate
VENV=$VENVS/dense_loader/bin/activate
SETUP_ENV=$CODE/scripts/bash/setup_new_env.sh
# LOAD_ENV=${WORK}/load_env.sh

# logs
LOG_DIR=${CODE}/scripts/logs/re10k_dense/debug_loader/min_env


# make dirs
mkdir -p ${LOG_DIR}

# messages
LOG_MSG="[SLURM][INFO]"
DURATION_MSG="$LOG_MSG[TIME] Duration:"
END_OF_JOB_MSG="${LOG_MSG} --- END OF JOB ---"

# main message
GENERAL_VARS=(TASK_NAME DESCRIPTION)
PARAM_VARS=(SCRIPT_PATH SCENE_PATH LOG_DIR VENV)
GENERAL_HEADER="------------------------- GENERAL --------------------------"
PARAM_HEADER="------------------------- PARAMETERS -------------------------"
END_HEADER="----------------------------------------------------------"
RUNNING_MSG="-------------------- RUNNING COMMAND ---------------------"
cat <<EOF
--- [SLURM SESSION] DATE: $(date) ---
USER: $(whoami)
NODE: $(hostname)
PATH: $(pwd)
${GENERAL_HEADER}
EOF
for var in "${GENERAL_VARS[@]}"; do
    printf "%s: %s\n" "$var" "${!var}"
done
echo -e "${PARAM_HEADER}"
for var in "${PARAM_VARS[@]}"; do
    printf "%s: %s\n" "$var" "${!var}"
done
echo -e "${END_HEADER}\n"


# --- jobs start - main process ---
SECONDS=0

echo -e "${LOG_MSG} Activating venv..."
# TORCH_LIB=$(python -c "import torch, pathlib; print(pathlib.Path(torch.__file__).parent/'lib')")
# echo "TORCH_LIB=$TORCH_LIB"
# export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"
# echo "HDF5_USE_FILE_LOCKING=FALSE"
# export HDF5_USE_FILE_LOCKING=FALSE
source $SETUP_ENV dense_loader_minimal "no_color" > /dev/null 2>&1
# source "$VENV"
# source "$LOAD_ENV"

cd ${SCRIPT_PATH}
echo -e "${LOG_MSG} Current Directory: $(pwd)"

echo -e "${LOG_MSG} Running loader for re10k..."

ARGS=(
    --ROOT ${SCENE_PATH} 
    --split "None"   
    --resolution 644
)

echo -e "${RUNNING_MSG}"
echo -e "python ${SCRIPT_PATH}/loader.py \n$(print_args ARGS)"
echo -e "${END_HEADER}\n"

python ${SCRIPT_PATH}/loader.py ${ARGS[@]}

if [ $? -eq 0 ]; then
    echo -e "\n$LOG_MSG [SUCCESS] '$TASK_NAME' completed successfully."
else
    echo -e "\n$LOG_MSG [ERROR] '$TASK_NAME' failed."
    exit 1
fi

DURATION=$SECONDS
echo -e "$DURATION_MSG $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds."
echo -e "$END_OF_JOB_MSG"
