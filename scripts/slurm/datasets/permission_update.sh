#!/bin/bash

#SBATCH --job-name=permission_update
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/permission_update/%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/permission_update/%j.err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060

# --- CONFIGURATION ---
TARGET_DIR="$SCRATCH/DL3DV960_slurm"
# ---------------------

echo "Starting permission update on: $TARGET_DIR"

# Apply 775 to all existing directories and 664 to files
chmod -R 775 "$TARGET_DIR"
find "$TARGET_DIR" -type f -exec chmod 664 {} +

# Set the SetGID bit so new files inherit the parent group ID
find "$TARGET_DIR" -type d -exec chmod g+s {} +

# Set Default ACLs so future files/folders inherit rwx for the group
setfacl -R -d -m g::rwx "$TARGET_DIR"

echo "Permissions updated successfully."
getfacl "$TARGET_DIR"