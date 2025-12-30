import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import List, Tuple
from pathlib import Path
from ..core.path import get_save_path, extract_wai_meta_paths, extract_dense_paths, get_unique_path
from ..core.util import subsample_paths

logger = logging.getLogger(__name__)


def get_plot_dimensions(models: List[str], frames: int, width_ratio=4, height_ratio=4) -> Tuple[int, int, int]:
    """
    Calculates figure dimensions and column count based on model count.

    Args:
        - models (List[str]): List of depth models to compare.
        - frames (int): Number of frames to display (rows).
        - width_ratio (float): Width ratio per column (default: 4).
        - height_ratio (float): Height ratio per row (default: 4).
    """
    num_models = len(models)
    total_cols = 1 + (num_models * 2)
    width = total_cols * width_ratio
    height = frames * height_ratio
    logger.info("Figure size:", extra={"width": width, "height": height, "total_cols": total_cols})

    return total_cols, width, height


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
        depth_map = np.load(depth_path)
    else:
        # exr and other formats
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth_map is None:
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    # handle extra channels, e.g., (H, W, C) - keep only the first channel if it's single-channel
    if len(depth_map.shape) > 2:
        logger.warning(f"Depth map has extra channels: {depth_map.shape}")
        depth_map = depth_map[:, :, 0]

    depth_map = np.nan_to_num(depth_map.astype(np.float32), posinf=0, neginf=0)

    # normalize to 0-1
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


def _build_model_sets(
    scene_root: str,
    models: List[str],
    dense_dir: str,
    skip_step: int = 0,
) -> Tuple[dict, str]:
    """
    Build dictionary of available models with their paths.

    Returns:
        Tuple of (sets dictionary, rgb_path_for_save)
    """
    rgb_path_for_save = None
    sets = {}

    for model in models:
        if model == "mvsanywhere":
            result = extract_wai_meta_paths(scene_root, "mvsanywhere_depth")
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                if rgb_path_for_save is None:
                    rgb_path_for_save = result["rgb_paths"][0]
                sets["mvsanywhere"] = result
        elif model == "moge2":
            result = extract_wai_meta_paths(scene_root, "moge2_depth")
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                if rgb_path_for_save is None:
                    rgb_path_for_save = result["rgb_paths"][0]
                sets["moge2"] = result
        elif model == "depthanything":
            result = extract_dense_paths(dense_dir, model)
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                sets["depthanything"] = result

    return sets, rgb_path_for_save


def _calculate_figure_size(
    models: List[str],
    num_frames: int,
    width_ratio: float,
    height_ratio: float,
    preserve_resolution: bool,
    img_width_px: int,
    img_height_px: int,
    dpi: int,
    wspace: float,
    hspace: float,
) -> Tuple[int, float, float]:
    """
    Calculate figure dimensions based on resolution preservation or ratios.

    Returns:
        Tuple of (total_cols, fig_width, fig_height)
    """
    total_cols, fig_width, fig_height = get_plot_dimensions(models, num_frames, width_ratio, height_ratio)

    # calculate figure size to maintain 1:1 pixel mapping
    if preserve_resolution and img_width_px and img_height_px:
        # Calculate figure size based on image dimensions and DPI
        # Each subplot should be img_width_px/dpi inches wide to maintain pixel-perfect display
        subplot_width_inches = img_width_px / dpi
        subplot_height_inches = img_height_px / dpi

        # total fig size = subplot size * number of subplots
        # account for spacing between subplots (wspace is fraction of subplot width)
        spacing_width = wspace * subplot_width_inches * (total_cols - 1) if total_cols > 1 else 0
        spacing_height = hspace * subplot_height_inches * (num_frames - 1) if num_frames > 1 else 0

        fig_width = subplot_width_inches * total_cols + spacing_width
        fig_height = subplot_height_inches * num_frames + spacing_height

        logger.info(
            "Figure size calculated from image resolution:",
            extra={
                "fig_width": fig_width,
                "fig_height": fig_height,
                "dpi": dpi,
                "subplot_size": f"{subplot_width_inches:.2f}x{subplot_height_inches:.2f} inches",
            },
        )

    return total_cols, fig_width, fig_height


def _process_frame_data(
    sets: dict,
    frame_idx: int,
    alpha: float,
) -> Tuple[np.ndarray, str, dict]:
    """
    Process RGB and depth data for a single frame.

    Returns:
        Tuple of (rgb_img, frame_title, plot_per_model dict)
    """
    rgb_img = None
    frame_title = ""
    plot_per_model = {}

    for model in sets:
        model_data = sets[model]
        rgb_path = model_data["rgb_paths"][frame_idx]
        frame_name = model_data["frame_names"][frame_idx]
        depth_path = model_data["depth_paths"][frame_idx]

        rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if rgb is None:
            logger.error(f"Error reading RGB: {rgb_path}")
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        rgb_shape = rgb.shape

        try:
            _, _, depth_colored = process_depth(depth_path, rgb_shape)
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Error processing depth: {depth_path} - {e}")
            continue

        overlay = cv2.addWeighted(rgb, 1 - alpha, depth_colored, alpha, 0)

        if rgb_img is None:
            rgb_img = rgb
            frame_title = frame_name

        plot_per_model[model] = {
            "depth_colored": depth_colored,
            "overlay": overlay,
        }

    return rgb_img, frame_title, plot_per_model


def _plot_frame_row(
    axes_row,
    rgb_img: np.ndarray,
    frame_title: str,
    plot_per_model: dict,
    sets: dict,
    alpha: float,
    show_titles: bool,
    title_fontsize: float,
):
    """
    Plot a single row (frame) in the comparison grid.
    """
    col_idx = 0

    # Plot RGB image
    ax = axes_row[col_idx]
    ax.imshow(rgb_img)
    if show_titles:
        ax.set_title(f"{frame_title} - RGB", fontsize=title_fontsize, pad=2)
    ax.axis("off")
    ax.set_aspect("auto")
    col_idx += 1

    # Plot each model's colorized depth
    for model in sets:
        ax = axes_row[col_idx]
        ax.imshow(plot_per_model[model]["depth_colored"])
        if show_titles:
            ax.set_title(f"{model} Map", fontsize=title_fontsize, pad=2)
        ax.axis("off")
        ax.set_aspect("auto")
        col_idx += 1

    # Plot each model's overlay
    for model in sets:
        ax = axes_row[col_idx]
        ax.imshow(plot_per_model[model]["overlay"])
        if show_titles:
            ax.set_title(f"{model} Overlay (A={alpha})", fontsize=title_fontsize, pad=2)
        ax.axis("off")
        ax.set_aspect("auto")
        col_idx += 1


def _apply_figure_layout(
    fig,
    show_titles: bool,
    wspace: float,
    hspace: float,
    comment: str,
):
    """
    Apply spacing, layout adjustments, and optional comment to the figure.
    """
    # Minimize all spacing - use tight_layout with minimal padding
    if show_titles:
        plt.tight_layout(pad=0.5, w_pad=wspace, h_pad=hspace)
        # Adjust margins to be very tight but allow for small titles
        plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005, wspace=wspace, hspace=hspace)
    else:
        plt.tight_layout(pad=0, w_pad=wspace, h_pad=hspace)
        # Absolute minimum margins when no titles
        plt.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0, wspace=wspace, hspace=hspace)

    if comment:
        fig.text(
            0.5,
            0.998,
            comment,
            fontsize=8,
            color="blue",
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="gainsboro", alpha=0.9),
        )


def _save_or_show_figure(
    fig,
    output_dir: str,
    rgb_path_for_save: str,
    dpi: int,
    pad_inches: float,
):
    """
    Save the figure to file or display it.
    """
    if output_dir:
        if rgb_path_for_save:
            out_file = get_save_path(rgb_path_for_save, output_dir)
        else:
            out_file = os.path.join(os.path.abspath(output_dir), "comparison_plot_compact.png")
        out_file = get_unique_path(Path(out_file))
        logger.info("Figure saved to:", extra={"path": str(out_file), "dpi": dpi})
        plt.savefig(out_file, bbox_inches="tight", pad_inches=pad_inches, dpi=dpi, facecolor="white")
    else:
        plt.show()
        logger.info("Displayed comparison plot - not saved")
    plt.close()


def _build_model_sets(
    scene_root: str,
    models: List[str],
    dense_dir: str,
    skip_step: int = 0,
) -> Tuple[dict, str]:
    """
    Build dictionary of available models with their paths.

    Returns:
        Tuple of (sets dictionary, rgb_path_for_save)
    """
    rgb_path_for_save = None
    sets = {}

    for model in models:
        if model == "mvsanywhere":
            result = extract_wai_meta_paths(scene_root, "mvsanywhere_depth")
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                if rgb_path_for_save is None:
                    rgb_path_for_save = result["rgb_paths"][0]
                sets["mvsanywhere"] = result
        elif model == "moge2":
            result = extract_wai_meta_paths(scene_root, "moge2_depth")
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                if rgb_path_for_save is None:
                    rgb_path_for_save = result["rgb_paths"][0]
                sets["moge2"] = result
        elif model == "depthanything":
            result = extract_dense_paths(dense_dir, model)
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                sets["depthanything"] = result

    return sets, rgb_path_for_save
