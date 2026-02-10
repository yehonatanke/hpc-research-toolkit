# !/bin/bash
# SBATCH --job-name=9K_add_dup_ext
# SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/handle_dup/%x_%j.out.log
# SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/handle_dup/%x_%j.err.log
# SBATCH --account=AIFAC_S02_060
# SBATCH --time=04:00:00
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=8
# SBATCH --partition=lrd_all_serial
# SBATCH --qos=normal
# SBATCH --mem=4G
# SBATCH --gres=tmpfs:200g

### FAST AND OPTIMIZED UNZIP | HANDLE DUPLICATES FOR THIS DATASET WHERE 'images_4' AND 'transforms.json' SOMETIMES HAVE NO PARENT FOLDER ###

# operate on the specified subdirectory, e.g. "11K" or "10K"
DL3DV960_UNZIP_SUBDIR="8K"


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

    # local dest_dir="$SCRATCH/DL3DV960_slurm/$DL3DV960_UNZIP_SUBDIR"
    local dest_dir="$WORK/data/dl3dv_960/$DL3DV960_UNZIP_SUBDIR"
    local src_dir="$SCRATCH/DL3DV960_unzipped/$DL3DV960_UNZIP_SUBDIR"

    if [ ! -d "$src_dir" ]; then
        echo "[ERROR] DIRECTORY '$src_dir' DOES NOT EXIST. ABORTING UNZIP."
        return 1
    fi

    # Ensure the destination category folder exists
    mkdir -p "$dest_dir"

    # Count number of zip files before extraction
    zip_count=$(find "$src_dir" -maxdepth 1 -type f -name '*.zip' | wc -l)
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) NUMBER OF ZIP FILES TO EXTRACT: $zip_count"

    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) FOLDER COUNTS IN DESTINATION ($dest_dir) BEFORE EXTRACTION:"
    _count_types_one_level "$dest_dir"

    extracted_count_before=$(find "$dest_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)

    # skips the loop if no zip files are found

    # Main loop
    for zip_path in "$src_dir"/*.zip ; do
        [ -e "$zip_path" ] || continue
        
        # 1. Get the root folder name inside the zip (assumes zip contains one root folder)
        # 'unzip -Z -1' lists files, 'head -n 1' takes the first, 'cut' gets the root dir
        root_folder=$(unzip -Z -1 "$zip_path" | head -n 1 | cut -d/ -f1)
        
        # Clean vars
        zip_filename=$(basename "$zip_path" .zip)
        target_path="$dest_dir/$root_folder"

        # 2. Check if destination already exists
        if [ -d "$target_path" ]; then
            # Collision detected!
            echo "[INFO] Collision found for $root_folder in $zip_filename. Renaming..."
            
            # Create a temp dir for extraction
            temp_extract_dir=$(mktemp -d)
            
            # Extract to temp
            unzip -q "$zip_path" -d "$temp_extract_dir"
            
            # Construct new unique name: _DUP_<ZipName>_<FolderName>
            # (Using ZipName ensures multiple duplicates don't overwrite each other)
            new_dirname="_DUP_${zip_filename}_${root_folder}"
            
            # Move and Rename to final dest
            mv "$temp_extract_dir/$root_folder" "$dest_dir/$new_dirname"
            
            # Cleanup
            rm -rf "$temp_extract_dir"
        else
            # No collision - normal extraction
            unzip -o -q "$zip_path" -d "$dest_dir"
        fi
    done

    extracted_count_after=$(find "$dest_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    new_folders=$((extracted_count_after - extracted_count_before))
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) EXTRACTED $new_folders NEW FOLDERS IN $dest_dir."
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) FOLDER COUNTS IN DESTINATION ($dest_dir) AFTER EXTRACTION:"
    _count_types_one_level "$dest_dir"
    echo ""
}

# use this 
unzip_optimized() {
    local dest_dir="$SCRATCH/DL3DV960_DUP_EXT/$DL3DV960_UNZIP_SUBDIR"
    local src_dir="$SCRATCH/DL3DV960_unzipped/$DL3DV960_UNZIP_SUBDIR"
    local temp_staging_dir="${dest_dir}_staging_temp"
    echo -e "\n--- STARTING UNZIP OPTIMIZED PROCESS ---"
    echo -e "TEMP STAGING DIR: ${temp_staging_dir}"
    mkdir -p "$dest_dir" "$temp_staging_dir"

    echo -e "\n--- EXPORTING FUNCTION FOR XARGS ---"
    export STAGING_DIR="$temp_staging_dir"
    _unzip_worker() {
        zip_path="$1"
        zip_name=$(basename "$zip_path" .zip)
        target="$STAGING_DIR/$zip_name"
        mkdir -p "$target"
        unzip -q -o "$zip_path" -d "$target"
    }
    export -f _unzip_worker

    echo -e "\n--- GETTING SAFE CORE COUNT ---"
    local job_limit=$(nproc 2>/dev/null || echo 4)
    echo -e "\n[INFO] STARTING PARALLEL EXTRACTION WITH $job_limit JOBS..."

    find "$src_dir" -name "*.zip" -print0 | xargs -0 -n 1 -P "$job_limit" bash -c '_unzip_worker "$@"' _

    echo -e "\n[INFO] EXTRACTION DONE. MERGING..."

    for zip_folder in "$temp_staging_dir"/*; do
        [ -d "$zip_folder" ] || continue
        
        zip_name=$(basename "$zip_folder")
        
        for content_path in "$zip_folder"/*; do
            content_name=$(basename "$content_path")
            
            # Logic: If special content name, force subdir. Else, unzip regularly (handle collisions).
            if [[ "$content_name" == "images_4" || "$content_name" == "transforms.json" ]]; then
                target_dir="$dest_dir/$zip_name"
                mkdir -p "$target_dir"
                mv "$content_path" "$target_dir/"
            else
                final_dest="$dest_dir/$content_name"
                if [ -e "$final_dest" ]; then
                    echo -e "\n[INFO] COLLISION FOUND: $content_name IN $zip_name"
                    target_dir="$dest_dir/$zip_name"
                    mkdir -p "$target_dir"
                    mv "$content_path" "$target_dir/"
                else
                    mv "$content_path" "$dest_dir/"
                fi
            fi
        done
    done

    rm -rf "$temp_staging_dir"
    echo "[INFO] DONE."
}

unzip_optimized_bugs() {
    local dest_dir="$SCRATCH/DL3DV960_DUP_EXT/$DL3DV960_UNZIP_SUBDIR"
    local src_dir="$SCRATCH/DL3DV960_unzipped/$DL3DV960_UNZIP_SUBDIR"
    local temp_staging_dir="${dest_dir}_staging_temp"  # Use sibling dir to ensure atomic move
    echo -e "--- STARTING UNZIP OPTIMIZED PROCESS ---"
    echo -e "TEMP STAGING DIR: ${temp_staging_dir}"
    mkdir -p "$dest_dir" "$temp_staging_dir"

    echo -e "--- EXPORTING FUNCTION FOR XARGS ---"
    # Export function for xargs
    export STAGING_DIR="$temp_staging_dir"
    _unzip_worker() {
        zip_path="$1"
        zip_name=$(basename "$zip_path" .zip)
        # Extract into: staging/zipname/content
        target="$STAGING_DIR/$zip_name"
        mkdir -p "$target"
        unzip -q -o "$zip_path" -d "$target"
    }
    export -f _unzip_worker

    echo -e "--- GETTING SAFE CORE COUNT ---"
    # Get safe core count (defaults to 4 if nproc fails)
    local job_limit=$(nproc 2>/dev/null || echo 4)
    echo "[INFO] STARTING PARALLEL EXTRACTION WITH $job_limit JOBS..."

    # -P "$job_limit" instead of -P 0
    find "$src_dir" -name "*.zip" -print0 | xargs -0 -n 1 -P "$job_limit" bash -c '_unzip_worker "$@"' _

    echo "[INFO] EXTRACTION DONE. MERGING..."

    # Sequential Merge (Fast move operation)
    for zip_folder in "$temp_staging_dir"/*; do
        [ -d "$zip_folder" ] || continue
        
        zip_name=$(basename "$zip_folder")
        
        # Inside each zip wrapper is the actual content folder(s)
        for content_path in "$zip_folder"/*; do
            content_name=$(basename "$content_path")
            final_dest="$dest_dir/$content_name"

            if [ -e "$final_dest" ]; then
                # Collision: Rename
                echo -e "\n[INFO] COLLISION FOUND: $content_name IN $zip_name"
                new_dest_path="${dest_dir}/${zip_name}/${content_name}"
                echo -e "NEW DESTINATION: ${new_dest_path}\n"
                mv "$content_path" "${new_dest_path}"
                # new_name="${content_name}_DUP_${zip_name}"
                # mv "$content_path" "$dest_dir/$new_name"
            else
                # No collision
                mv "$content_path" "$dest_dir/"
            fi
        done
    done

    rm -rf "$temp_staging_dir"
    echo "[INFO] DONE."
}

# unzip_dl3dv960_category_folders
unzip_optimized