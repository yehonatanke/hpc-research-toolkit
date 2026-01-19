_print_job_header() {
local GENERAL_HEADER="------------------------ GENERAL -------------------------"
local PARAM_HEADER="------------------------- PARAMETERS -------------------------"
local END_HEADER="----------------------------------------------------------"

cat <<EOF
--- [Slurm] DATE: $(date) ---
USER: $(whoami)
NODE: $(hostname)
PATH: $(pwd)
$GENERAL_HEADER
EOF
    for var in "${GENERAL_VARS[@]}"; do
        printf "%s: %s\n" "$var" "${!var:-NOT_SET}"
    done
    echo "$PARAM_HEADER"
    for var in "${PARAM_VARS[@]}"; do
        printf "%s: %s\n" "$var" "${!var:-NOT_SET}"
    done
    echo -e "$END_HEADER\n"
}

print_duration() {
    echo -e "$DURATION_MSG $(($DURATION / 60)) minutes and $(($DURATION % 60)) seconds."
}