#!/bin/bash

### USAGE: SPLIT DATASET FOR SYNCING

# CONFIGURATIONS
SUBSET="8K"
SRC="${SCRATCH}/DL3DV960_DUP_EXT/${SUBSET}"
DEST="${WORK}/data/dl3dv_960/${SUBSET}"
LIST_DIR="${CODE}/scripts/slurm/datasets/dl3dv/sync_split_lists/${SUBSET}"

mkdir -p "$DEST" "$LIST_DIR"

# Generate full file list relative to SRC
find "$SRC" -maxdepth 1 -mindepth 1 -printf "%f\n" > "${LIST_DIR}/full_list.txt"

# Split into two equal parts
total_files=$(wc -l < "${LIST_DIR}/full_list.txt")
half_mark=$(( (total_files + 1) / 2 ))

split -l "$half_mark" -d "${LIST_DIR}/full_list.txt" "${LIST_DIR}/part_"
