YAML_FILE="${CODE}/scripts/logs/datasets/dl3dv/hashes/intersections.yaml"
SRC_ROOT="${WORK}/data/dl3dv_960"
DEST_ROOT="${WORK}/data/dl3dv_960/removed_benchmarks"

echo "Moving intersecting folders to ${DEST_ROOT}..."

declare -i SUM
current_subset=""

while IFS= read -r line; do
    # Detect Subset Header (e.g., "1K:")
    if [[ "$line" =~ ^([0-9]+K):$ ]]; then
        current_subset="${BASH_REMATCH[1]}"
        mkdir -p "${DEST_ROOT}/${current_subset}"
        echo "PROCESSING SUBSET: ${current_subset}"

    elif [[ "$line" =~ ^[[:space:]]*-[[:space:]]*(.+)$ ]]; then
        [ -z "$current_subset" ] && continue
        
        hash_name="${BASH_REMATCH[1]}"
        src_path="${SRC_ROOT}/${current_subset}/${hash_name}"
        dest_path="${DEST_ROOT}/${current_subset}/"

        if [ -d "$src_path" ]; then
            echo "  MOVING ${hash_name}..."
            mv "$src_path" "$dest_path"
            SUM+=1
        else
            echo "  [WARNING]: ${src_path} NOT FOUND."
        fi
    fi
done < "$YAML_FILE"

echo -e "SUM: ${SUM}"
echo "DONE."
unset SUM