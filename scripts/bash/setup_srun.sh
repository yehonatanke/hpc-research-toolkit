#!/bin/bash

# Usage: ./setup_srun.sh [HH:MM:SS]

# If no argument is provided, defaults to 01:00:00
TIME_ARG="${1:-00:30:00}"

TITLE_PARAMS=("=== COMMAND EXECUTED ===" "\033[38;5;197m")
_title_bg_soft_white "${TITLE_PARAMS[0]}"
echo -e "srun --partition=boost_usr_prod \
     \n\t--qos=normal \
     \n\t--nodes=1 \
     \n\t--ntasks=1 \
     \n\t--cpus-per-task=8 \
     \n\t--gres=gpu:1 \
     \n\t--time="$TIME_ARG" \
     \n\t--account=AIFAC_S02_060 \
     \n\t--pty /bin/bash"


srun --partition=boost_usr_prod \
     --qos=normal \
     --nodes=1 \
     --ntasks=1 \
     --cpus-per-task=8 \
     --gres=gpu:1 \
     --time="$TIME_ARG" \
     --account=AIFAC_S02_060 \
     --pty /bin/bash


unset TIME_ARG