# Production script to unzip files directly into category folders.
# Since the .zip files already contain the scene-named folder, 
# we unzip into DL3DV960/<subdir> to avoid double-nesting.

# Prompt the user to confirm running unzip_dl3dv960_category_folders in the current directory
current_dir="$(pwd)"
echo "You are about to run 'unzip_dl3dv960_category_folders' in: $current_dir"
read -p "Do you want to proceed? (y/n): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborting script as per user request."
    exit 0
fi

unzip_dl3dv960_category_folders() {
    if [ ! -d "DL3DV960_unzipped" ]; then
        echo "[Error] Directory 'DL3DV960_unzipped' does not exist. Aborting unzip."
        return 1
    fi

    for dir in DL3DV960_unzipped/*/ ; do
        subdir=$(basename "$dir")
        dest="DL3DV960/$subdir"

        # Ensure the destination category folder (e.g., DL3DV960/11K) exists
        mkdir -p "$dest"

        # Count number of zip files before extraction
        zip_count=$(find "$dir" -maxdepth 1 -type f -name '*.zip' | wc -l)
        echo "[INFO] ($subdir) Number of zip files to extract: $zip_count"

        echo "[INFO] ($subdir) Folder counts in destination ($dest) BEFORE extraction:"
        _count_types_one_level "$dest"

        extracted_count_before=$(find "$dest" -mindepth 1 -maxdepth 1 -type d | wc -l)

        # skips the loop if no zip files are found
        for zip_path in "$dir"*.zip ; do
            [ -e "$zip_path" ] || continue

            # Extracting into the category folder uses the folder structure inside the ZIP
            unzip -q "$zip_path" -d "$dest"
        done

        extracted_count_after=$(find "$dest" -mindepth 1 -maxdepth 1 -type d | wc -l)
        new_folders=$((extracted_count_after - extracted_count_before))
        echo "[INFO] ($subdir) Extracted $new_folders new folders in $dest."
        echo "[INFO] ($subdir) Folder counts in destination ($dest) AFTER extraction:"
        _count_types_one_level "$dest"
        echo ""
    done
}

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

# Call the function after user confirms
unzip_dl3dv960_category_folders
