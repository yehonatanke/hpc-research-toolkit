DIR="${CODE}/scripts/logs/datasets/dl3dv/hashes"
BENCHMARK="${DIR}/benchmark.txt"
OUTPUT="${DIR}/intersections.yaml"
declare -i SUM

echo "WRITING INTERSECTIONS TO ${OUTPUT}..."
: > "$OUTPUT"

for i in {1..11}; do
    SUBSET="${i}K"
    FILE="${DIR}/${SUBSET}.txt"
    
    if [ ! -f "$FILE" ]; then
        echo -e "[FILE NOT FOUND] SKIPPING: ${SUBSET}"
        continue
    fi

    echo -n "PROCESSING: ${SUBSET}... "

    echo "${SUBSET}:" >> "$OUTPUT"
    
    # Store matches in variable to count them before writing
    matches=$(awk 'NR==FNR{a[$0];next} $0 in a' "$BENCHMARK" "$FILE")
    
    if [ -n "$matches" ]; then
        echo "$matches" | sed 's/^/  - /' >> "$OUTPUT"
        count=$(echo "$matches" | wc -l)
        SUM+=count
        echo "FOUND [ ${count} ] MATCHES."
    else
        echo "NO MATCHES."
    fi
done

echo -e "SUM: ${SUM}"
unset SUM
echo "DONE."


# : > "$OUTPUT"

# for i in {1..11}; do
#     SUBSET="${i}K"
#     FILE="${DIR}/${SUBSET}.txt"
#     [ -f "$FILE" ] || continue

#     echo "${SUBSET}:" >> "$OUTPUT"
    
#     # Find intersection and format as YAML list
#     awk 'NR==FNR{a[$0];next} $0 in a' "$BENCHMARK" "$FILE" | \
#     sed 's/^/  - /' >> "$OUTPUT"
# done