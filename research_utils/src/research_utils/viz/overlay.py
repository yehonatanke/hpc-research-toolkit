import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

from research_utils import get_rgb_image, get_depth_map, get_save_path, ensure_dir, extract_wai_meta_paths, extract_dense_paths, subsample_plot_paths_list, subsample_paths

# from ..io.image import get_rgb_image, get_depth_map
# from ..core.path import get_save_path

logger = logging.getLogger(__name__)


def plot_overlay(rgb_path, depth_path, save_dir, model_name, alpha=0.6, comment: str = ""):
    if comment:
        comment = f"\nComment: {comment}"
    else:
        comment = ""

    rgb_img = get_rgb_image(rgb_path)
    depth_map = get_depth_map(depth_path)

    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_normalized = (depth_map - d_min) / (d_max - d_min + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    logger.info(
        "RGB & Depth Dimentions:",
        extra={"RGB": rgb_img.shape[:2], "Depth": depth_colored.shape[:2]},
    )

    if rgb_img.shape[:2] != depth_colored.shape[:2]:
        rgb_orig = rgb_img.shape[:2]
        depth_orig = depth_colored.shape[:2]
        depth_colored = cv2.resize(depth_colored, (rgb_img.shape[1], rgb_img.shape[0]))
        logger.warning(
            "Resizing depth to match RGB dimensions",
            extra={f"RGB: {rgb_orig} --> {rgb_img.shape[:2]}", f"Depth: {depth_orig} --> {depth_colored.shape[:2]}"},
        )

    overlay = cv2.addWeighted(rgb_img, 1 - alpha, depth_colored, alpha, 0)

    plt.figure(figsize=(15, 5))
    images = [rgb_img, depth_colored, overlay]
    titles = ["RGB", "Depth", f"Overlay ({alpha})"]

    if model_name:
        plt.gcf().text(
            0.02,
            0.96,
            f"Model: {model_name}{comment}",
            fontsize=10,
            color="blue",
            horizontalalignment="left",
            verticalalignment="top",
            bbox=dict(facecolor="gainsboro", alpha=0.5, edgecolor="none", boxstyle="round"),
        )

    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.title(title, fontsize=10, color="black")
        plt.axis("off")
        plt.tight_layout()

    plt.tight_layout()
    out_file = get_save_path(rgb_path, save_dir)
    logger.info("Figure saved to:", extra={"path": out_file})

    plt.savefig(out_file)
    plt.close()

    return out_file


####
# os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # enable openexr support

from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from pathlib import Path

# from modules.util import extract_meta_paths, get_filename, read_rgb, get_unique_path
# from modules.logger import debug_print



def process_depth(depth_path: str, rgb_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes a single depth map from either .npy, .exr, or common image formats.
    Returns:
        - depth_map: raw depth map
        - depth_normalized: normalized depth (0-1)
        - depth_colored: colorized depth (RGB)
    """
    ext = os.path.splitext(depth_path)[1].lower()
    if ext == ".npy":
        # load numpy depth
        depth_map = np.load(depth_path)
    else:
        # exr and other image formats can be read via OpenCV
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_map is None:
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    # handle extra channels, e.g., (H, W, C) - keep only the first channel if it's single-channel
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    # convert to float32 for later computation (OpenEXR may provide float32/float16)
    depth_map = np.nan_to_num(depth_map.astype(np.float32), posinf=0, neginf=0)

    # normalize depth map to 0-1 range
    valid_mask = (depth_map > 0) & np.isfinite(depth_map)
    if np.any(valid_mask):
        d_min = depth_map[valid_mask].min()
        d_max = depth_map[valid_mask].max()
    else:
        d_min = 0
        d_max = 1

    # Avoid division by zero and handle constant/empty depth maps
    if d_max == d_min:
        depth_normalized = np.zeros_like(depth_map, dtype=np.float32)
    else:
        depth_normalized = (depth_map - d_min) / (d_max - d_min)
        # mask invalid pixels to 0 in normalized map
        depth_normalized[~valid_mask] = 0

    # normalize to 0-255 range for colorization
    depth_uint8 = (np.clip(depth_normalized, 0, 1) * 255).astype(np.uint8)

    # colorize depth map
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    if depth_colored.shape[:2] != rgb_shape[:2]:
        logger.warning(f"Resizing depth map from {depth_colored.shape} to {rgb_shape}")
        depth_colored = cv2.resize(depth_colored, (rgb_shape[1], rgb_shape[0]))

    return depth_map, depth_normalized, depth_colored


# def get_fig_size(models: List[str], rgb_paths: List[str], width_ratio: float = 4, height_ratio: float = 4):
#     num_models = len(models)
#     total_cols = 1 + (2 * num_models)
#     total_rows = len(rgb_paths)
#     fig_width = total_cols * width_ratio
#     fig_height = total_rows * height_ratio
#     logger.info("Figure size:", extra={"width": fig_width, "height": fig_height})

#     return fig_width, fig_height


def get_plot_dimensions(models: List[str], frames: int, base_ratio=4)->Tuple[int, int, int]:
    """
    Calculates figure dimensions and column count based on model count.
    
    Args:
        - models (List[str]): List of depth models to compare.
        - frames (int): Number of frames to display (rows).
        - base_ratio (int): Plotting units per individual subplot.
    """
    num_models = len(models)
    total_cols = 1 + (num_models * 2)
    width = total_cols * base_ratio
    height = frames * base_ratio
    logger.info("Figure size:", extra={"width": width, "height": height, "total_cols": total_cols})
    return total_cols, width, height

def plot_comparison_multiple(
    scene_root: str,
    models: List[str],
    dense_dir: str,
    output_dir: str,
    alpha: float = 0.6,
    skip_step: int = 0,
    width_ratio: float = 4,
    height_ratio: float = 4,
    comment: str = "",
):
    """
    Args:
        scene_root: wai scene root directory
        models: list of depth models to compare
        dense_dir: directory containing the dense depth maps
        output_dir: directory to save the output comparison plots
        alpha: alpha value for the overlay
        skip_step: step size to skip frames (default: 0)
        width_ratio: width ratio of the figure (default: 4)
        height_ratio: height ratio of the figure (default: 4)
        comment: comment to add to the output plot (default: "")
    """
    if output_dir:
        ensure_dir(output_dir)
    
    filename =""

    # Build a dictionary of available models, extracting only those in `models`
    sets = {}
    for model in models:
        if model == "mvsanywhere":
            result = extract_wai_meta_paths(scene_root, "mvsanywhere_depth")
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                filename = os.path.basename(result["rgb_paths"][0])
                sets["mvsanywhere"] = result
        elif model == "moge2":
            result = extract_wai_meta_paths(scene_root, "moge2_depth")
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                filename = os.path.basename(result["rgb_paths"][0])
                sets["moge2"] = result
        elif model == "mapanything":
            result = extract_dense_paths(dense_dir, model)
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                sets["mapanything"] = result

    non_empty_lengths = {k: len(v.get("rgb_paths", [])) for k, v in sets.items() if v and v.get("rgb_paths")}
    if not non_empty_lengths:
        logger.error("No valid depth maps found for any model in {models}")
        return

    # any available integer value (e.g., the first)
    num_frames = next(iter(non_empty_lengths.values()), 0)

    total_cols, fig_width, fig_height = get_plot_dimensions(models, num_frames, width_ratio, height_ratio)

    # Handle single row indexing edge case
    if num_frames == 1: 
        axes = axes.reshape(1, -1)

    logger.info("Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height})
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid.")
    plt.figure(figsize=(fig_width, fig_height))

    # In alignment with the construction of `sets`, which maps model names to their info dicts
    from collections import OrderedDict

    # Only keep models present in `sets` and keep input order
    filtered_models = [m for m in models if m in sets and sets[m].get("rgb_paths")]
    filtered_sets = OrderedDict((m, sets[m]) for m in filtered_models)
    num_models = len(filtered_sets)

    total_rows = num_frames  

    for i in tqdm(range(num_frames)):
        base_index = i * total_cols

        rgb_img = None
        frame_title = ""
        plot_per_model = {}

        for model in filtered_models:
            model_data = filtered_sets[model]
            rgb_path = model_data["rgb_paths"][i]
            frame_name = model_data["frame_names"][i]
            depth_path = model_data["depth_paths"][i]

            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                logger.error(f"Error reading RGB: {rgb_path}")
                continue 
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_shape = rgb.shape

            _, _, depth_colored = process_depth(depth_path, rgb_shape)
            overlay = cv2.addWeighted(rgb, 1 - alpha, depth_colored, alpha, 0)

            if rgb_img is None:
                rgb_img = rgb
                frame_title = frame_name

            plot_per_model[model] = {
                "depth_colored": depth_colored,
                "overlay": overlay,
            }

        subplot_idx = base_index + 1
        plt.subplot(total_rows, total_cols, subplot_idx)
        plt.imshow(rgb_img)
        plt.title(f"{frame_title} - RGB")
        plt.axis("off")
        subplot_idx += 1

        # Plot each model's colorized depth
        for model in filtered_models:
            plt.subplot(total_rows, total_cols, subplot_idx)
            plt.imshow(plot_per_model[model]["depth_colored"])
            plt.title(f"{model} Map")
            plt.axis("off")
            subplot_idx += 1

        # Plot each model's overlay
        for model in filtered_models:
            plt.subplot(total_rows, total_cols, subplot_idx)
            plt.imshow(plot_per_model[model]["overlay"])
            plt.title(f"{model} Overlay (A={alpha})")
            plt.axis("off")
            subplot_idx += 1

    plt.tight_layout()

    if output_dir:
        out_file = get_save_path(filename, output_dir)
        logger.info("Figure saved to:", extra={"path": out_file})
        plt.savefig(out_file)
    else:
        plt.show()
        logger.info("Displayed comparison plot - not saved")
    plt.close()


def plot_overlay_one(rgb_path, depth_path, output_dir, alpha=0.6):
    # load rgb image
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        debug_print(1, f"Error reading RGB: {rgb_path}")
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # convert to matplotlib format

    # load depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        debug_print(1, f"Error reading Depth: {depth_path}")
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

    d_min, d_max = depth_map.min(), depth_map.max()
    depth_normalized = (depth_map - d_min) / (d_max - d_min + 1e-8)  # avoid division by 0
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
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(depth_colored)
    plt.title("Depth Map (Colorized)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title(f"Overlay (Alpha={alpha})")
    plt.axis("off")

    filename = os.path.basename(os.path.dirname(os.path.dirname(rgb_path))) + "_" + os.path.basename(rgb_path)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved overlay to {os.path.join(output_dir, filename)}")
