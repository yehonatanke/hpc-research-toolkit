# Aesthetic variables
RESET='\[\e[0m\]'       # reset all attributes
BOLD='\[\e[1m\]'        
WHITE='\[\e[97m\]'      # bright white
GRAY='\[\e[90m\]'       # light gray
CYAN='\[\e[96m\]'       # bright cyan
YELLOW='\[\e[93m\]'     # bright yellow
GREEN='\[\e[92m\]'      # bright green
NEWLINE=$'\n'
DARK_WHITE='\033[0;37m' # for github branch
GREEN_ANSI='\e[38;5;46m'
GREEN_ANSI_2='\e[38;5;83m'
GIT_RED='\033[0;31m'
GIT_GREEN='\033[0;32m' # darker
GIT_YELLOW='\033[0;33m'
GIT_BLUE='\033[0;34m'
GIT_MAGENTA='\033[0;35m'
GIT_CYAN='\033[0;36m'
GIT_GRAY='\033[0;90m'
GIT_BOLD='\033[1m'
GIT_RESET='\033[0m'
SEP_DEFAULT=' | '
SEP_DOT=' ∙ '

# vars
export BASHRC="$HOME/.bashrc"

# dirs
export YK="$WORK/data/yk"
export CODE="$WORK/data/yk/code"
export DEBUG="$WORK/data/yk/debug"
export DEPTH_OVERLAY="$CODE/apps/analysis/depth_overlay"
export DIR_SLURM="$DEBUG/scripts/slurm"
export DUMMY_DL3DV="$DEBUG/dl3dv_wai_dummy"
export REPOS="$YK/repos"
export MAP_ANYTHING="$REPOS/map-anything"


# dummy sata & samples
export DUMMY="$DEBUG/dl3dv_wai_dummy"
export DUMMY_1="$DUMMY/1K_0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3"
export DUMMY_2="$DUMMY/1K_00b1ad87c296635c73bbc9728a63d3df4a9ab07aee0d34cd45784e39f2c699ba"
export DUMMY_3="$DUMMY/3K_9e249fd778184f593bc51e3b63fe27fa6cef1cb897a2d51aeba813c59bc63724"
export DUMMY_4="$DUMMY/4K_0ed0f1c119cb25ae7b01c095f7ab4bab3c6d81ebfaac961709a89b45837a68d7"
export DUMMY_5="$DUMMY/4K_5536979395665d8edb7ea6fe0f35d4df9b43fe5a751668b164d0d12140728677"
export DUMMY_6="$DUMMY/4K_98724539e82458e819ef575fb66747c28305844a5db4841e08ca85d9d564beba"

# dense
export DENSE="$DEBUG/output/da3/dense"

# output dirs
export OUT_DENSE_OVERLAY="$DEPTH_OVERLAY/output"

# envs
export VENVS="$YK/envs"
export VENV_VISER="$YK/envs/viser-env/bin/activate"
export VENV_MAPA="$YK/envs/map-anything-venv/bin/activate"
export VENV_DA3="$YK/envs/depth-anything-env/bin/activate"
export VENV_ANALYSIS="$CODE/.venvs/analysis-venv/bin/activate"



# --- Aliases ---

# activate venvs
alias activate_viser="source $VENV_VISER"
alias activate_ma="source $VENV_MAPA" # activate map-anything venv
alias activate_da3="source $VENV_DA3"
alias activate_analysis="source $VENV_ANALYSIS"

# set venvs and go to directories
alias set_ma="activate_ma && goto_mapa" # activate map-anything venv & go to map-anything dir

# go to directories
alias goto_slurm="cd $DIR_SLURM" # go to slurm dir
alias goto_mapa="cd $MAP_ANYTHING" # go to map-anything dir
alias goto_venvs="cd $VENVS"
alias goto_dummy_dl3dv="cd $DUMMY_DL3DV"
alias goto_code="cd $CODE" 

# open and load .bashrc
alias open_bashrc="vi $BASHRC"
alias load_bashrc="source $BASHRC"