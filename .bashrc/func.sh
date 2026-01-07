# count files & dirs in a directory
count_dir() {
    local target="${1:-.}"
    if [ ! -d "$target" ]; then
        echo "Directory not found: $target" >&2
        return 1
    fi

    local files=$(find "$target" -maxdepth 1 -type f | wc -l)
    local dirs=$(find "$target" -maxdepth 1 -type d | wc -l)
    local total=$(find "$target" -maxdepth 1 | wc -l)

    printf "Files: %'d\n" "$files"
    printf "Dirs:  %'d\n" "$((dirs - 1))" # Excludes the target directory itself
    printf "Total: %'d\n" "$((total - 1))"
}

# usage: print_args ARGS
print_args() {
    local -n _arr=$1
    local i=0
    while (( i < ${#_arr[@]} )); do
        if [[ $((i + 1)) -lt ${#_arr[@]} ]] && [[ "${_arr[i+1]}" != --* ]]; then
            printf "%s %s\n\t" "${_arr[i]}" "${_arr[i+1]}"
            ((i += 2))
        else
            printf "%s\n\t" "${_arr[i]}"
            ((i++))
        fi
    done
}

# without tab
print_args_v1() {
    local -n _arr=$1
    local i=0
    while (( i < ${#_arr[@]} )); do
        if [[ $((i + 1)) -lt ${#_arr[@]} ]] && [[ "${_arr[i+1]}" != --* ]]; then
            printf "%s %s\n" "${_arr[i]}" "${_arr[i+1]}"
            ((i += 2))
        else
            printf "%s\n" "${_arr[i]}"
            ((i++))
        fi
    done
}

run_depth_overlay() {
    # repo root
    local REPO_DIR="$DEPTH_OVERLAY"
    
    # save current dir and move to repo
    pushd "$REPO_DIR" > /dev/null || return
    
    python -m scripts.run_depth_overlay "$@"
    
    # Return to previous dir
    popd > /dev/null
}

run_depth_compare() {
    # repo root
    local REPO_DIR="$DEPTH_OVERLAY_COMPARE"
    
    # save current dir and move to repo
    pushd "$REPO_DIR" > /dev/null || return
    
    python -m scripts.compare_overlay "$@"
    
    # Return to previous dir
    popd > /dev/null
}

# sbatch wrapper 
submit() {

    local ACCOUNT="AIFAC_S02_060"
    local DEBUG="$WORK/data/yk/debug"
    sbatch --account=$ACCOUNT \
           --output=$DEBUG/output/%j.out \
           --error=$DEBUG/logs/%j.err \
           "$@"
}

# print the shape of a numpy array
npyshape() {
    python3 -c "import numpy as np; import sys; print(np.load(sys.argv[1], mmap_mode='r').shape)" "$1"
}

countfiles() {
    local target="${1:-.}"
    if [ -d "$target" ]; then
        expr $(ls -f "$target" | wc -l) - 2
    else
        echo "Directory not found"
        return 1
    fi
}

# print the statistics of numpy arrays in a folder
npy_stats() {
    python3 -c "
import numpy as np
import os
import sys
from collections import Counter

folder = sys.argv[1]
shapes = []
skipped = 0
total_files = 0

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    if not os.path.isfile(path):
        continue
    total_files += 1
    if f.endswith('.npy'):
        try:
            data = np.load(path, mmap_mode='r')
            shapes.append(data.shape)
        except Exception:
            skipped += 1
    else:
        skipped += 1

stats = Counter(shapes)
for shape, count in sorted(stats.items()):
    print(f'- resolution: {shape}: {count}')

print(f'\nTotal files in folder: {total_files}')
print(f'Files skipped: {skipped}')
" "$1"
}


# print the statistics of images in a folder
img_stats() {
    python3 -c "
import os
import sys
from collections import Counter
from PIL import Image

folder = sys.argv[1]
shapes = []
skipped = 0
total_files = 0
extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    if not os.path.isfile(path):
        continue
    total_files += 1
    if f.lower().endswith(extensions):
        try:
            with Image.open(path) as img:
                shapes.append(img.size)
        except Exception:
            skipped += 1
    else:
        skipped += 1

stats = Counter(shapes)
for (w, h), count in sorted(stats.items()):
    print(f'- resolution: ({h}, {w}): {count}')

print(f'\nTotal files in folder: {total_files}')
print(f'Files skipped: {skipped}')
" "$1"
}

# print the statistics of npz files in a folder
npz_stats() {
    python3 -c "
import numpy as np
import os
import sys
from collections import Counter

folder = sys.argv[1]
shapes = []
skipped = 0
total_files = 0

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    if not os.path.isfile(path):
        continue
    total_files += 1
    if f.endswith('.npz'):
        try:
            with np.load(path) as data:
                for key in data.files:
                    shapes.append(data[key].shape)
        except Exception:
            skipped += 1
    else:
        skipped += 1

print(f'\n- Total files in folder: {total_files}')
stats = Counter(shapes)
for shape, count in sorted(stats.items()):
    print(f'- resolution: {shape}: {count}')

print(f'\nFiles skipped (non-npz or corrupt): {skipped}')
" "$1"
}

# print the statistics of files in a folder - npz, npy, images, etc.
file_stats() {
    python3 -c "
import numpy as np
import os
import sys
from collections import Counter
from PIL import Image

folder = sys.argv[1]
shapes = []
skipped = 0
total_files = 0
detected_types = set()

img_exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff')

for f in os.listdir(folder):
    path = os.path.join(folder, f)
    if not os.path.isfile(path):
        continue
    
    total_files += 1
    ext = os.path.splitext(f)[1].lower()
    
    try:
        if ext == '.npy':
            data = np.load(path, mmap_mode='r')
            shapes.append(data.shape)
            detected_types.add('NPY')
        elif ext == '.npz':
            with np.load(path) as data:
                for key in data.files:
                    shapes.append(data[key].shape)
            detected_types.add('NPZ')
        elif ext in img_exts:
            with Image.open(path) as img:
                # Standardizing to (H, W) or (H, W, C) to match numpy style
                w, h = img.size
                mode = img.mode
                channels = len(mode) if mode != 'P' else 3
                shapes.append((h, w, channels) if channels > 1 else (h, w))
            detected_types.add('Image')
        else:
            skipped += 1
    except Exception:
        skipped += 1

stats = Counter(shapes)
print(f'Detected Types: {\", \".join(detected_types) if detected_types else \"None\"}')
for shape, count in sorted(stats.items()):
    print(f'- resolution: {shape}: {count}')

print(f'\nTotal files in folder: {total_files}')
print(f'Files skipped: {skipped}')
" "$1"
}

# print png as matrix
pngmat() {
    python3 -c "import PIL.Image, numpy as np; import sys; print(np.array(PIL.Image.open(sys.argv[1])))" "$1"
}

# view matrix of a file - npy, npz, or image
viewmat() {
    python3 -c "
import sys, numpy as np, PIL.Image
path = sys.argv[1]
if path.endswith('.npy'):
    data = np.load(path)
elif path.endswith('.npz'):
    data = np.load(path); print({k: data[k].shape for k in data.files}); [print(f'{k}:\n', data[k]) for k in data.files]; sys.exit()
else:
    data = np.array(PIL.Image.open(path))
print(data)
" "$1"
}