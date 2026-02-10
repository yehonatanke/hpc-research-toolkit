#!/bin/bash
#SBATCH --job-name=9K
#SBATCH --output=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/handle_dup/%x_%j.out.log
#SBATCH --error=/leonardo_work/AIFAC_S02_060/data/yk/code/scripts/logs/datasets/unzip_dl3dv_960/handle_dup/%x_%j.err.log
#SBATCH --account=AIFAC_S02_060
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=lrd_all_serial
#SBATCH --qos=normal
#SBATCH --mem=4G
#SBATCH --gres=tmpfs:200g


# operate on the specified subdirectory, e.g. "11K" or "10K"
DL3DV960_UNZIP_SUBDIR="9K"

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

export TMPDIR="${SLURM_TMPDIR:-/tmp}"

process_single_zip() {
    local zip_path="$1"
    local dest_dir="$2"
    local zip_name
    zip_name=$(basename "$zip_path" .zip)
    local target_path="$dest_dir/$zip_name"
    
    local temp_stage
    temp_stage=$(mktemp -d)

    if ! unzip -q "$zip_path" -d "$temp_stage"; then
         echo "[WARNING] Corrupt zip skipped: $zip_name"
         rm -rf "$temp_stage"
         return
    fi

    mkdir -p "$target_path"

    # Fast path: Move non-colliding files
    rsync -a --recursive --ignore-existing --remove-source-files "$temp_stage/" "$target_path/"

    # Slow path: Handle collisions
    if [ -n "$(ls -A "$temp_stage")" ]; then
        find "$temp_stage" -type f | while read -r src_file; do
            local rel_path="${src_file#$temp_stage/}"
            local dest_file="$target_path/$rel_path"
            
            local extension="${dest_file##*.}"
            local filename="${dest_file%.*}"
            local timestamp=$(date +%s%N)
            local new_dest="${filename}_DUP_${timestamp}.${extension}"
            
            mkdir -p "$(dirname "$new_dest")"
            mv "$src_file" "$new_dest"
        done
    fi

    rm -rf "$temp_stage"

    # --- PROGRESS TRACKING ---
    # Append a line to the shared counter file
    echo "1" >> "$PROGRESS_TRACKER_FILE"
    # Count lines to get current status
    local current_count=$(wc -l < "$PROGRESS_TRACKER_FILE")
    # Calculate percentage (integer math)
    local percent=$(( (current_count * 100) / TOTAL_ZIPS_COUNT ))
    
    echo "[PROGRESS] ${current_count}/${TOTAL_ZIPS_COUNT} (${percent}%) - Finished: $zip_name"
}

export -f process_single_zip

# use this 
unzip_handle_dup_optimization() {
    # local dest_dir="$WORK/data/dl3dv_960/$DL3DV960_UNZIP_SUBDIR"
    local dest_dir="$SCRATCH/DL3DV960_HANDLE_DUP/$DL3DV960_UNZIP_SUBDIR"
    local src_dir="$SCRATCH/DL3DV960_unzipped/$DL3DV960_UNZIP_SUBDIR"

    if [ ! -d "$src_dir" ]; then
        echo "[ERROR] DIRECTORY '$src_dir' DOES NOT EXIST."
        return 1
    fi

    mkdir -p "$dest_dir"
    
    # 1. Initialize Global Counters for Subshells
    export TOTAL_ZIPS_COUNT=$(find "$src_dir" -maxdepth 1 -name '*.zip' | wc -l)
    export PROGRESS_TRACKER_FILE=$(mktemp) # Create temp file for IPC

    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) STARTING PARALLEL EXTRACTION OF $TOTAL_ZIPS_COUNT FILES..."

    # 2. Run Parallel Extraction
    find "$src_dir" -maxdepth 1 -name '*.zip' -print0 | \
        xargs -0 -P "${SLURM_CPUS_PER_TASK:-1}" -n 1 -I {} \
        bash -c 'process_single_zip "$@"' _ {} "$dest_dir"

    # 3. Cleanup
    # rm -f "$PROGRESS_TRACKER_FILE"
    echo -e "PROGRESS_TRACKER_FILE: $PROGRESS_TRACKER_FILE"

    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) COMPLETE."
    _count_types_one_level "$dest_dir"
}


unzip_handle_dup_no_optimization() {
    # This function takes zip files from "$SCRATCH/DL3DV960_unzipped_slurm/$DL3DV960_UNZIP_SUBDIR"
    # and unzips them into "$SCRATCH/DL3DV960/$DL3DV960_UNZIP_SUBDIR/<ZIP_FILENAME>/"
    
    local dest_dir="$WORK/data/dl3dv_960/$DL3DV960_UNZIP_SUBDIR"
    local src_dir="$SCRATCH/DL3DV960_unzipped/$DL3DV960_UNZIP_SUBDIR"

    if [ ! -d "$src_dir" ]; then
        echo "[ERROR] DIRECTORY '$src_dir' DOES NOT EXIST. ABORTING UNZIP."
        return 1
    fi

    mkdir -p "$dest_dir"

    # Count number of zip files before extraction
    zip_count=$(find "$src_dir" -maxdepth 1 -type f -name '*.zip' | wc -l)
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) NUMBER OF ZIP FILES TO EXTRACT: $zip_count"

    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) FOLDER COUNTS IN DESTINATION ($dest_dir) BEFORE EXTRACTION:"
    _count_types_one_level "$dest_dir"

    extracted_count_before=$(find "$dest_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)

    for zip_path in "$src_dir"/*.zip ; do
        [ -e "$zip_path" ] || continue

        zip_name=$(basename "$zip_path" .zip)
        target_path="$dest_dir/$zip_name"
        
        # Create a temporary staging area
        temp_stage=$(mktemp -d)

        # Unzip into temp dir (clean extraction, no conflicts yet)
        if ! unzip -q "$zip_path" -d "$temp_stage"; then
             echo "[WARNING] Corrupt zip skipped: $zip_name"
             rm -rf "$temp_stage"
             continue
        fi

        # Ensure target root exists
        mkdir -p "$target_path"

        # Move files from temp to target with collision checks
        # Using find to handle recursive structures properly
        find "$temp_stage" -type f | while read -r src_file; do
            # Calculate relative path (e.g., "images/01.jpg")
            rel_path="${src_file#$temp_stage/}"
            dest_file="$target_path/$rel_path"
            dest_dir_parent=$(dirname "$dest_file")

            mkdir -p "$dest_dir_parent"

            if [ -e "$dest_file" ]; then
                # --- COLLISION HANDLING ---
                echo "[DUPLICATE FOUND] Processing: $rel_path in $zip_name"
                
                # Generate new name: filename_DUP_<timestamp>.ext
                extension="${dest_file##*.}"
                filename="${dest_file%.*}"
                timestamp=$(date +%s%N) # Nanoseconds for uniqueness
                new_dest="${filename}_DUP_${timestamp}.${extension}"
                
                echo "   -> Saving duplicate as: $(basename "$new_dest")"
                mv "$src_file" "$new_dest"
            else
                # No collision, just move
                mv "$src_file" "$dest_file"
            fi
        done

        # Cleanup temp dir
        rm -rf "$temp_stage"
    done

    extracted_count_after=$(find "$dest_dir" -mindepth 1 -maxdepth 1 -type d | wc -l)
    new_folders=$((extracted_count_after - extracted_count_before))
    
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) EXTRACTED $new_folders NEW PARENT FOLDERS IN $dest_dir."
    echo "[INFO] ($DL3DV960_UNZIP_SUBDIR) FOLDER COUNTS IN DESTINATION ($dest_dir) AFTER EXTRACTION:"
    _count_types_one_level "$dest_dir"
    echo ""
}

unzip_handle_dup_optimization