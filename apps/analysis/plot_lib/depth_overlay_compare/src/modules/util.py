import argparse
import json
from pathlib import Path
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # enable openexr support

import cv2
from modules.logger import debug_print


def print_args(args):
    print(f"\n===========[INFO] ARGS:===========")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")
    print(f"===================================\n")


def extract_meta_paths(scene_meta_path, depths_to_compare=["mvsanywhere_depth", "moge2_depth"]):
    """
    Extracts depths from scene_meta.json
    returns:
        - rgb_paths: list of rgb paths
        - frame_names: list of frame names
        - depth_paths_1: list of mvsanywhere_depth paths
        - depth_paths_2: list of moge2_depth paths
    """
    debug_print(1, f"Extracting depth paths for {depths_to_compare}")
    rgb_paths = []
    frame_names = []
    depth_paths_1 = []
    depth_paths_2 = []

    with open(scene_meta_path, 'r') as f:
        data = json.load(f)
    
    for frame in data.get('frames', []):
        rgb_paths.append(frame.get('image'))
        frame_names.append(frame.get('frame_name'))
        depth_1 = frame.get(depths_to_compare[0])
        depth_2 = frame.get(depths_to_compare[1])

        if depth_1:
            depth_paths_1.append(depth_1)
        if depth_2:
            depth_paths_2.append(depth_2)

    debug_print(1, f"Extracted {len(rgb_paths)} rgb paths")
    debug_print(1, f"Extracted {len(depth_paths_1)} depth paths for {depths_to_compare[0]}")
    debug_print(1, f"Extracted {len(depth_paths_2)} depth paths for {depths_to_compare[1]}")

    return rgb_paths, frame_names, depth_paths_1, depth_paths_2


def get_filename(rgb_path):
    """
    get filename from rgb path
    returns:
        - filename: filename of the rgb image
    """
    return os.path.basename(os.path.dirname(os.path.dirname(rgb_path))) + "_" + os.path.basename(rgb_path).replace(os.path.splitext(rgb_path)[1], ".png")


def read_rgb(rgb_path):
    """
    read rgb image
    returns:
        - rgb_img: rgb image
    """
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        debug_print(1, f"Error reading RGB: {rgb_path}.")
        return
    return rgb_img


def get_unique_path(filepath: Path) -> Path:
    """
    Returns a unique Path object. If the original path exists,
    it appends a counter (e.g., _1, _2) before the file extension.
    """
    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent
    
    counter = 1
    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_filepath = parent / new_filename
        
        if not new_filepath.exists():
            return new_filepath
        
        counter += 1