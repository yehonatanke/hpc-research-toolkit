### ---------------------- Dry run script: ---------------------- ###
# Dry run script to mirror the directory structure of DL3DV960_unzipped into DL3DV960.
# ----------------------------------------------------------------------------


# It iterates through each subdirectory, processes up to 10 .zip files as a sample,
# and prints the unzip command that would create a folder named after the scene.
# (!) Do NOT assume each zip already contains a scene-named folder.

# Dry run showing unzip with scene folder destination
dry_run_unzip_scene_folder() {
    for dir in DL3DV960_unzipped/*/ ; do
        subdir=$(basename "$dir")
        echo "--- Processing Subdirectory: $subdir ---"

        count=0
        for zip_path in "$dir"*.zip ; do
            [ -e "$zip_path" ] || continue

            # Stop after 10 files per subdirectory
            if [ $count -eq 10 ]; then
                echo "Skipping remaining files in $subdir..."
                break
            fi

            scene_name=$(basename "$zip_path" .zip)
            dest="DL3DV960/$subdir/$scene_name"

            # Dry run: printing the command instead of executing
            echo "[DRY RUN] unzip -q \"$zip_path\" -d \"$dest\""

            ((count++))
        done
        echo ""
    done
}

### ------------------------------------------------------------ ###

# --- Unzip files into category folders ---
# (!) Assumes each zip already contains a scene-named folder to avoid double-nesting.
# ----------------------------------------------------------------------------
# .
# └── DL3DV960
#     ├── 1K/
#     │   └── SCENE_01/
#     │       └── [zip contents]
#     └── 11K/
#         └── SCENE_99/
#             └── [zip contents]
# ----------------------------------------------------------------------------

# Dry run showing unzip into category folder only
dry_run_unzip_category_folder() {
    for dir in DL3DV960_unzipped/*/ ; do
        subdir=$(basename "$dir")
        echo "--- Checking Subdirectory: $subdir ---"

        count=0
        for zip_path in "$dir"*.zip ; do
            [ -e "$zip_path" ] || continue

            if [ $count -eq 10 ]; then
                echo "Skipping remaining files in $subdir..."
                break
            fi

            # Destination is the category folder (e.g., DL3DV960/1K)
            # Zip content (e.g., SCENE_01/) will be placed inside it.
            dest="DL3DV960/$subdir"

            echo "[DRY RUN] unzip -q \"$zip_path\" -d \"$dest\""

            ((count++))
        done
        echo ""
    done
}

# Example calls (comment/uncomment as desired)
# dry_run_unzip_scene_folder
# dry_run_unzip_category_folder