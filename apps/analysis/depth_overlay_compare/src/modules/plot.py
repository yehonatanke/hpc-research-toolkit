import os 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # enable openexr support

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from pathlib import Path
from modules.util import extract_meta_paths, get_filename, read_rgb, get_unique_path
from modules.logger import debug_print


def process_depth(depth_path: str, rgb_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    process a single depth map
    returns:
        - depth_map: raw depth map
        - depth_normalized: normalized depth (0-1)
        - depth_colored: colorized depth (RGB)
    """
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    # handle extra channels if any
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    # replace NaN/Inf with 0
    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

    # normalize depth map to 0-1 range
    d_min, d_max = depth_map.min(), depth_map.max()
    # Avoid division by zero and handle constant depth maps
    if d_max == d_min:
        depth_normalized = np.zeros_like(depth_map, dtype=np.float32)
    else:
        depth_normalized = (depth_map - d_min) / (d_max - d_min)

    # normalize to 0-255 range for colorization
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # colorize depth map
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    if depth_colored.shape != rgb_shape:
        debug_print(1, f"Resizing depth map from {depth_colored.shape} to {rgb_shape}")
        depth_colored = cv2.resize(depth_colored, (rgb_shape[1], rgb_shape[0]))

    return depth_map, depth_normalized, depth_colored

def plot_comparison_multiple(
    scene_root: str,
    depths_to_compare: List[str],
    output_dir: str,
    alpha: float = 0.6,
    skip_step: int = 0,
    width_ratio: float = 4,
    height_ratio: float = 4,
    save: bool = False
):
    """
    Args:
        scene_root: root directory of the scene
        depths_to_compare: list of depth map paths to compare
        output_dir: directory to save the output comparison plots
        alpha: alpha value for the overlay
        skip_step: step size to skip frames
        width_ratio: width ratio of the figure
        height_ratio: height ratio of the figure
        save: if True, save output images
    """
    if save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            debug_print(1, f"Created save directory: {output_dir}")
        else:
            debug_print(1, f"Save directory already exists: {output_dir}")

    scene_meta_path = os.path.join(scene_root, "scene_meta.json")
    rgb_paths, frame_names, d1_paths, d2_paths = extract_meta_paths(scene_meta_path, depths_to_compare)
    d1_name = depths_to_compare[0]
    d2_name = depths_to_compare[1]

    if skip_step > 0 and skip_step < len(rgb_paths):
        rgb_paths = rgb_paths[::skip_step]
        frame_names = frame_names[::skip_step]
        d1_paths = d1_paths[::skip_step]
        d2_paths = d2_paths[::skip_step]
        debug_print(1, f"Skipping every {skip_step} frames")

    # determine figure size
    num_sets = len(frame_names)
    rows_per_plot_set = 1
    total_cols = 5 

    total_rows = num_sets * rows_per_plot_set
    
    fig_width = total_cols * width_ratio
    fig_height = total_rows * height_ratio 

    plt.figure(figsize=(fig_width, fig_height))
    debug_print(1, f"Plotting {num_sets} comparison sets on a {total_rows}x{total_cols} grid.")

    # for i, (rgb_path, frame_name, d1_path, d2_path) in enumerate(zip(rgb_paths, frame_names, d1_paths, d2_paths)):
    for i, (rgb_path, frame_name, d1_path, d2_path) in enumerate(tqdm(zip(rgb_paths, frame_names, d1_paths, d2_paths))):
        try:
            base_index = i * total_cols
            
            rgb_path = os.path.join(scene_root, rgb_path)
            d1_path = os.path.join(scene_root, d1_path)
            d2_path = os.path.join(scene_root, d2_path)

            rgb_img = read_rgb(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_shape = rgb_img.shape

            _, _, depth_1_colored = process_depth(d1_path, rgb_shape)
            _, _, depth_2_colored = process_depth(d2_path, rgb_shape)

            overlay_1 = cv2.addWeighted(rgb_img, 1 - alpha, depth_1_colored, alpha, 0)
            overlay_2 = cv2.addWeighted(rgb_img, 1 - alpha, depth_2_colored, alpha, 0)

            # subplot 1: Original RGB
            plt.subplot(total_rows, total_cols, base_index + 1)
            plt.imshow(rgb_img)
            plt.title(f"{frame_name} - RGB")
            plt.axis('off')

            # subplot 2: Depth Map 1 (Colorized)
            plt.subplot(total_rows, total_cols, base_index + 2)
            plt.imshow(depth_1_colored)
            plt.title(f"{d1_name} Map")
            plt.axis('off')

            # subplot 3: Depth Map 2 (Colorized)
            plt.subplot(total_rows, total_cols, base_index + 3)
            plt.imshow(depth_2_colored)
            plt.title(f"{d2_name} Map")
            plt.axis('off')

            # subplot 4: Overlay 1
            plt.subplot(total_rows, total_cols, base_index + 4)
            plt.imshow(overlay_1)
            plt.title(f"{d1_name} Overlay (A={alpha})")
            plt.axis('off')

            # subplot 5: Overlay 2
            plt.subplot(total_rows, total_cols, base_index + 5)
            plt.imshow(overlay_2)
            plt.title(f"{d2_name} Overlay (A={alpha})")
            plt.axis('off')

        except Exception as e:
            debug_print(1, f"[ERROR] An error occurred for {rgb_path}: {e} Skipping subplot.")
            raise e
    
    plt.tight_layout()

    if save:
        # filename = get_filename(rgb_path)
        # full_output_dir = os.path.join(output_dir, filename)
        # plt.savefig(full_output_dir)
        filename = get_filename(rgb_path)
        output_path = Path(output_dir) / filename
        unique_path = get_unique_path(output_path)
        plt.savefig(unique_path)
        debug_print(1, f"Saved comparison plot to {unique_path}")
    else:
        plt.show()
        debug_print(1, f"Displayed comparison plot - not saved")
    plt.close()


def plot_overlay_one(rgb_path, depth_path, output_dir, alpha=0.6):
    # load rgb image
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        debug_print(1, f"Error reading RGB: {rgb_path}")
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB) # convert to matplotlib format

    # load depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        debug_print(1, f"Error reading Depth: {depth_path}")
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")
    
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]
        
    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

    d_min, d_max = depth_map.min(), depth_map.max()
    depth_normalized = (depth_map - d_min) / (d_max - d_min + 1e-8) # avoid division by 0
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    if rgb_img.shape != depth_colored.shape:
        debug_print(1, f"Resizing depth from {depth_colored.shape} to {rgb_img.shape}")
        depth_colored = cv2.resize(depth_colored, (rgb_img.shape[1], rgb_img.shape[0]))

    overlay = cv2.addWeighted(rgb_img, 1 - alpha, depth_colored, alpha, 0)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_img)
    plt.title("Original RGB")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(depth_colored)
    plt.title("Depth Map (Colorized)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay (Alpha={alpha})")
    plt.axis('off')

    filename = os.path.basename(os.path.dirname(os.path.dirname(rgb_path))) + "_" + os.path.basename(rgb_path)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename)) 
    plt.close()
    print(f"Saved overlay to {os.path.join(output_dir, filename)}")
