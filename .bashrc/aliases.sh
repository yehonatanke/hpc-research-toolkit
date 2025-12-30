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
export ACCOUNT="AIFAC_S02_060"

# dirs
export YK="$WORK/data/yk"
export CODE="$WORK/data/yk/code"
export DEBUG="$WORK/data/yk/debug"
export DEPTH_OVERLAY="$CODE/apps/analysis/depth_overlay"
export DEPTH_OVERLAY_COMPARE="$CODE/apps/analysis/depth_overlay_compare"
export DIR_SLURM="$DEBUG/scripts/slurm"
export DUMMY_DL3DV="$DEBUG/dl3dv_wai_dummy"
export REPOS="$YK/repos"
export MAP_ANYTHING="$REPOS/map-anything"


# dummy sata & samples
export SAMPLE_1="1K_0a1b7c20a92c43c6b8954b1ac909fb2f0fa8b2997b80604bc8bbec80a1cb2da3"
export SAMPLE_2="1K_00b1ad87c296635c73bbc9728a63d3df4a9ab07aee0d34cd45784e39f2c699ba"
export SAMPLE_3="3K_9e249fd778184f593bc51e3b63fe27fa6cef1cb897a2d51aeba813c59bc63724"
export SAMPLE_4="4K_0ed0f1c119cb25ae7b01c095f7ab4bab3c6d81ebfaac961709a89b45837a68d7"
export SAMPLE_5="4K_5536979395665d8edb7ea6fe0f35d4df9b43fe5a751668b164d0d12140728677"
export SAMPLE_6="4K_98724539e82458e819ef575fb66747c28305844a5db4841e08ca85d9d564beba"
# wai
export WAI_DUMMY="$DEBUG/dl3dv_wai_dummy"
export WAI_DUMMY_1="$WAI_DUMMY/$SAMPLE_1"
export WAI_DUMMY_2="$WAI_DUMMY/$SAMPLE_2"
export WAI_DUMMY_3="$WAI_DUMMY/$SAMPLE_3"
export WAI_DUMMY_4="$WAI_DUMMY/$SAMPLE_4"
export WAI_DUMMY_5="$WAI_DUMMY/$SAMPLE_5"
export WAI_DUMMY_6="$WAI_DUMMY/$SAMPLE_6"
#da3
export DA3_DUMMY="$DEBUG/da3_dummy/dense"
export DA3_DUMMY_1="$DA3_DUMMY/$SAMPLE_1"
export DA3_DUMMY_2="$DA3_DUMMY/$SAMPLE_2"
export DA3_DUMMY_3="$DA3_DUMMY/$SAMPLE_3"
export DA3_DUMMY_4="$DA3_DUMMY/$SAMPLE_4"
export DA3_DUMMY_5="$DA3_DUMMY/$SAMPLE_5"
export DA3_DUMMY_6="$DA3_DUMMY/$SAMPLE_6"

# dense
export DENSE="$DEBUG/output/da3/dense"

# output dirs
export DENSE_OVERLAY_OUT="$DEPTH_OVERLAY/output"

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

alias gacp='f() { git add . && git commit -m "$1" && git push -u origin main; }; f'