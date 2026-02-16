# used to get all the hashes for dl3dv

list_filenames() {
    local target_dir="${1%/}"
    local output_file="$2"

    for entry in "$target_dir"/*; do
        [ -e "$entry" ] || continue
        echo "${entry##*/}"
    done > "$output_file"
}

# Usage: 
# list_filenames </path/to/dir> <results.txt>


### CONFIGS ###
TARGET_ROOT="${WORK}/data/dl3dv_960" 
OUT_ROOT="${CODE}/scripts/logs/datasets/dl3dv/hashes"

# Loop 1K to 11K
for i in {1..11}; do
    SUBSET="${i}K"
    
    TARGET="${TARGET_ROOT}/${SUBSET}"
    OUT="${OUT_ROOT}/${SUBSET}.txt"
    echo -e "\n[INFO] SUBSET: ${SUBSET} \n\t- TARGET_ROOT: ${TARGET_ROOT} \n\t- OUT_ROOT:${OUT_ROOT}"

    # Ensure output directory exists
    mkdir -p "${OUT%/*}"
    
    list_filenames "$TARGET" "$OUT"
    echo -e "[SUCCESS] SUBSET: ${SUBSET}"
done