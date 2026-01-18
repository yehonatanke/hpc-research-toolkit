#!/bin/bash

#SBATCH --job-name=Unzip_DL3DV960_11K
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/11K_%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/11K_%j.err.log
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --account=AIFAC_S02_060


# operate on the specified subdirectory, e.g. "11K" or "10K"
DL3DV960_UNZIP_SUBDIR="11K"


_count_types_one_level() {
    find "${1:-.}" -maxdepth 1 -mindepth 1 -printf '%y %f\n' | awk '
        $1 == "d" {a["directory"]++; next}
        $2 !~ /\./ {a["no_extension"]++; next}
        {
            n = split($0, parts, ".")
            ext = parts[n]
            a[ext]++
        }
        END {
            for (i in a) printf "%d %s\n", a[i], i
        }
    ' | sort -nr
}

_count_file_types_recursive() {
    find "${1:-.}" -type f -name "*.*" | awk -F. '{print $NF}' | sort | uniq -c | sort -nr
}

# main 
unzip_dl3dv960_category_folders() {
    # This function takes zip files from "$SCRATCH/DL3DV960_unzipped_slurm/$DL3DV960_UNZIP_SUBDIR"
    # and unzips them into "$SCRATCH/DL3DV960/$DL3DV960_UNZIP_SUBDIR".
    #
    # Example:
    #   - Source (zip input):      $SCRATCH/DL3DV960_unzipped/3K/[*.zip]
    #   - Destination (output):    $SCRATCH/DL3DV960_slurm/3K/

    local dest_dir="$SCRATCH/DL3DV960_slurm/$DL3DV960_UNZIP_SUBDIR"
    local src_dir="$SCRATCH/DL3DV960_unzipped/$DL3DV960_UNZIP_SUBDIR"

    if [ ! -d "$src_dir" ]; then
        echo "[Error] Directory '$src_dir' does not exist. Aborting unzip."
        return 1
    fi

    # Ensure the destination category folder exists
    mkdir -p "$dest_dir"

    # Count number of zip files before extraction
    zip_count=$(find "$src_dir" -maxdepth 1 -type f -name '*.zip' | wc -l)
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) Number of zip files to extract: $zip_count"

    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) Folder counts in destination ($dest_dir) BEFORE extraction:"
    _count_types_one_level "$dest_dir"

    extracted_count_before=$(find "$dest_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)

    # skips the loop if no zip files are found
    for zip_path in "$src_dir"/*.zip ; do
        [ -e "$zip_path" ] || continue

        # Extracting into the category folder uses the folder structure inside the ZIP
        unzip -q "$zip_path" -d "$dest_dir"
    done

    extracted_count_after=$(find "$dest_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    new_folders=$((extracted_count_after - extracted_count_before))
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) Extracted $new_folders new folders in $dest_dir."
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) Folder counts in destination ($dest_dir) AFTER extraction:"
    _count_types_one_level "$dest_dir"
    echo ""
}


unzip_dl3dv960_category_folders
