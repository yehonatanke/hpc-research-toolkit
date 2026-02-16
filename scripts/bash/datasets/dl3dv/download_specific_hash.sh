

VENV="${ENVS}/hf_env/bin/activate"
HASH="e78f8cebd2bd93d960bfaeac18fac0bb2524f15c44288903cd20b73e599e8a81"
SUBSET="8K"
SCRIPT="${CODE}/scripts/slurm/datasets/dl3dv/download.py"
OUT_DIR="${SCRATCH}/TEMP_CHECK_2"

mkdir -p ${OUT_DIR}
source $VENV

python $SCRIPT --odir $OUT_DIR --subset $SUBSET --resolution 960P --file_type images+poses --hash $HASH 
