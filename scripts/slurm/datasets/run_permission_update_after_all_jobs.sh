### THIS RUNS THE PERMISSION UPDATE SCRIPT AFTER ALL JOBS ARE COMPLETED ###

### CONFIGURATION ###
SLURM_JOB="${CODE}/scripts/slurm/datasets/permission_update.sh"

JOB_IDS=$(squeue -u $USER -h -o %i | tr '\n' ',' | sed 's/,$//')
echo -e "### JOB IDs ###\n${JOB_IDS}\n"

sbatch --dependency=afterany:$JOB_IDS $SLURM_JOB
echo -e "### RUNNING PERMISSION UPDATE SCRIPT AFTER ALL JOBS ARE COMPLETED ###"
