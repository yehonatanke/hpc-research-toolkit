import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from typing import List, Tuple
from ..io.image import get_rgb_image, get_depth_map
from pathlib import Path
from ..core.path import get_save_path, ensure_dir, extract_wai_meta_paths, extract_dense_paths, get_unique_path
from ..core.util import subsample_paths

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
    out_file = get_unique_path(Path(out_file))
    logger.info("Figure saved to:", extra={"path": str(out_file)})

    plt.savefig(out_file)
    plt.close()

    return out_file


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


def get_plot_dimensions(models: List[str], frames: int, width_ratio=4, height_ratio=4)->Tuple[int, int, int]:
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
    
    rgb_path_for_save = None

    # Build a dictionary of available models, extracting only those in `models`
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

    logger.info("Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height})
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid.")
    plt.figure(figsize=(fig_width, fig_height))
    # The filtered_models/filtered_sets block has been removed per instructions.
    # Use the input models and sets directly; ensure models exist in sets and have valid "rgb_paths" later if needed.

    total_rows = num_frames

    # plot_depth_comparison_grid(
    #     sets,
    #     models,
    #     num_frames,
    #     alpha,
    #     output_dir,
    #     scene_root,
    #     comment
    #     )
    # return None

    for i in tqdm(range(num_frames)):
        base_index = i * total_cols

        rgb_img = None
        frame_title = ""
        plot_per_model = {}

        for model in sets:
            model_data = sets[model]
            rgb_path = model_data["rgb_paths"][i]
            frame_name = model_data["frame_names"][i]
            depth_path = model_data["depth_paths"][i]

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

        subplot_idx = base_index + 1
        plt.subplot(total_rows, total_cols, subplot_idx)
        plt.imshow(rgb_img)
        plt.title(f"{frame_title} - RGB")
        plt.axis("off")
        subplot_idx += 1

        # Plot each model's colorized depth
        for model in sets:
            plt.subplot(total_rows, total_cols, subplot_idx)
            plt.imshow(plot_per_model[model]["depth_colored"])
            plt.title(f"{model} Map")
            plt.axis("off")
            subplot_idx += 1

        # Plot each model's overlay
        for model in sets:
            plt.subplot(total_rows, total_cols, subplot_idx)
            plt.imshow(plot_per_model[model]["overlay"])
            plt.title(f"{model} Overlay (A={alpha})")
            plt.axis("off")
            subplot_idx += 1

    plt.tight_layout()

    if output_dir:
        if rgb_path_for_save:
            out_file = get_save_path(rgb_path_for_save, output_dir)
        else:
            # Fallback: construct path directly if no RGB path available
            out_file = os.path.join(os.path.abspath(output_dir), "comparison_plot.png")
        out_file = get_unique_path(Path(out_file))
        logger.info("Figure saved to:", extra={"path": str(out_file)})
        plt.savefig(out_file)
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
        elif model == "mapanything":
            result = extract_dense_paths(dense_dir, model)
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                sets["mapanything"] = result
    
    return sets, rgb_path_for_save


def _get_image_dimensions(sets: dict) -> Tuple[int, int]:
    """
    Get image dimensions from the first available image.
    
    Returns:
        Tuple of (img_width_px, img_height_px) or (None, None) if not found
    """
    if not sets:
        return None, None
    
    first_model = next(iter(sets.keys()))
    first_rgb_path = sets[first_model]["rgb_paths"][0]
    first_img = cv2.imread(first_rgb_path, cv2.IMREAD_COLOR)
    
    if first_img is not None:
        img_height_px, img_width_px = first_img.shape[:2]
        logger.info("Image dimensions detected:", extra={"width": img_width_px, "height": img_height_px})
        return img_width_px, img_height_px
    
    return None, None


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
    
    # If preserving resolution, calculate figure size to maintain 1:1 pixel mapping
    if preserve_resolution and img_width_px and img_height_px:
        # Calculate figure size based on image dimensions and DPI
        # Each subplot should be img_width_px/dpi inches wide to maintain pixel-perfect display
        subplot_width_inches = img_width_px / dpi
        subplot_height_inches = img_height_px / dpi
        
        # Total figure size = subplot size * number of subplots
        # Account for spacing between subplots (wspace is fraction of subplot width)
        spacing_width = wspace * subplot_width_inches * (total_cols - 1) if total_cols > 1 else 0
        spacing_height = hspace * subplot_height_inches * (num_frames - 1) if num_frames > 1 else 0
        
        fig_width = subplot_width_inches * total_cols + spacing_width
        fig_height = subplot_height_inches * num_frames + spacing_height
        
        logger.info("Figure size calculated from image resolution:", 
                   extra={"fig_width": fig_width, "fig_height": fig_height, "dpi": dpi,
                          "subplot_size": f"{subplot_width_inches:.2f}x{subplot_height_inches:.2f} inches"})
    
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
    ax.set_aspect('auto')
    col_idx += 1

    # Plot each model's colorized depth
    for model in sets:
        ax = axes_row[col_idx]
        ax.imshow(plot_per_model[model]["depth_colored"])
        if show_titles:
            ax.set_title(f"{model} Map", fontsize=title_fontsize, pad=2)
        ax.axis("off")
        ax.set_aspect('auto')
        col_idx += 1

    # Plot each model's overlay
    for model in sets:
        ax = axes_row[col_idx]
        ax.imshow(plot_per_model[model]["overlay"])
        if show_titles:
            ax.set_title(f"{model} Overlay (A={alpha})", fontsize=title_fontsize, pad=2)
        ax.axis("off")
        ax.set_aspect('auto')
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
    
    # Add comment if provided (positioned to not interfere with subplots)
    if comment:
        fig.text(0.5, 0.998, comment, fontsize=8, ha='center', va='top', 
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))


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
            # Fallback: construct path directly if no RGB path available
            out_file = os.path.join(os.path.abspath(output_dir), "comparison_plot_compact.png")
        out_file = get_unique_path(Path(out_file))
        logger.info("Figure saved to:", extra={"path": str(out_file), "dpi": dpi})
        # Use bbox_inches='tight' with zero padding for minimal white space
        # High DPI preserves image resolution for quality model comparison
        plt.savefig(out_file, bbox_inches='tight', pad_inches=pad_inches, dpi=dpi, facecolor='white')
    else:
        plt.show()
        logger.info("Displayed comparison plot - not saved")
    plt.close()


def plot_comparison_multiple_compact(
    scene_root: str,
    models: List[str],
    dense_dir: str,
    output_dir: str,
    alpha: float = 0.6,
    skip_step: int = 0,
    width_ratio: float = 4,
    height_ratio: float = 4,
    comment: str = "",
    wspace: float = 0.0,
    hspace: float = 0.0,
    pad_inches: float = 0.0,
    show_titles: bool = True,
    title_fontsize: float = 7,
    dpi: int = 200,
    preserve_resolution: bool = True,
):
    """
    Improved version of plot_comparison_multiple_compact_white_space with minimal white space between subplots.
    Preserves image resolution.
    
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
        wspace: width space between subplots as fraction of average axes width (default: 0.0)
        hspace: height space between subplots as fraction of average axes height (default: 0.0)
        pad_inches: padding in inches around the figure when saving (default: 0.0)
        show_titles: whether to show subplot titles (default: True)
        title_fontsize: font size for titles (default: 7)
        dpi: resolution in dots per inch for saved figure (default: 200, higher = better quality)
        preserve_resolution: if True, calculate figure size from actual image dimensions (default: True)
    """
    if output_dir:
        ensure_dir(output_dir)
    
    # Build model sets
    sets, rgb_path_for_save = _build_model_sets(scene_root, models, dense_dir, skip_step)
    
    # Validate sets
    non_empty_lengths = {k: len(v.get("rgb_paths", [])) for k, v in sets.items() if v and v.get("rgb_paths")}
    if not non_empty_lengths:
        logger.error("No valid depth maps found for any model in {models}")
        return
    
    num_frames = next(iter(non_empty_lengths.values()), 0)
    
    # Get image dimensions for resolution preservation
    img_width_px, img_height_px = None, None
    if preserve_resolution:
        img_width_px, img_height_px = _get_image_dimensions(sets)
    
    # Calculate figure size
    total_cols, fig_width, fig_height = _calculate_figure_size(
        models, num_frames, width_ratio, height_ratio,
        preserve_resolution, img_width_px, img_height_px, dpi, wspace, hspace
    )
    
    logger.info("Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height})
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid with minimal spacing (DPI: {dpi}).")
    
    # Create figure and axes grid
    fig, axes = plt.subplots(num_frames, total_cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Handle single row case
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    # Process and plot each frame
    for i in tqdm(range(num_frames)):
        rgb_img, frame_title, plot_per_model = _process_frame_data(sets, i, alpha)
        _plot_frame_row(axes[i], rgb_img, frame_title, plot_per_model, sets, alpha, show_titles, title_fontsize)
    
    # Apply layout and spacing
    _apply_figure_layout(fig, show_titles, wspace, hspace, comment)
    
    # Save or show figure
    _save_or_show_figure(fig, output_dir, rgb_path_for_save, dpi, pad_inches)


def plot_overlay_one(rgb_path, depth_path, output_dir, alpha=0.6):
    # load rgb image
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        logger.error(f"Error reading RGB: {rgb_path}")
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # convert to matplotlib format

    # load depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        logger.error(f"Error reading Depth: {depth_path}")
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
        logger.warning(f"Resizing depth from {depth_colored.shape} to {rgb_img.shape}" ,extra={"rgb_path": rgb_path, "depth_path": depth_path})
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
    save_path = os.path.join(output_dir, filename)
    save_path = get_unique_path(Path(save_path))
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved overlay to {save_path}")


def plot_comparison_multiple_compact_white_space(
    scene_root: str,
    models: List[str],
    dense_dir: str,
    output_dir: str,
    alpha: float = 0.6,
    skip_step: int = 0,
    width_ratio: float = 4,
    height_ratio: float = 4,
    comment: str = "",
    wspace: float = 0.05,
    hspace: float = 0.05,
    pad_inches: float = 0.0,
):
    """
    Improved version of plot_comparison_multiple with white space between subplots.
    
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
        wspace: width space between subplots as fraction of average axes width (default: 0.05)
        hspace: height space between subplots as fraction of average axes height (default: 0.05)
        pad_inches: padding in inches around the figure when saving (default: 0.0)
    """
    if output_dir:
        ensure_dir(output_dir)
    
    rgb_path_for_save = None

    # Build a dictionary of available models, extracting only those in `models`
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

    logger.info("Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height})
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid with compact spacing.")
    
    # Use subplots for better control over spacing
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    total_rows = num_frames

    for i in tqdm(range(num_frames)):
        base_index = i * total_cols

        rgb_img = None
        frame_title = ""
        plot_per_model = {}

        for model in sets:
            model_data = sets[model]
            rgb_path = model_data["rgb_paths"][i]
            frame_name = model_data["frame_names"][i]
            depth_path = model_data["depth_paths"][i]

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

        subplot_idx = base_index + 1
        ax = plt.subplot(total_rows, total_cols, subplot_idx)
        ax.imshow(rgb_img)
        ax.set_title(f"{frame_title} - RGB", fontsize=9)
        ax.axis("off")
        subplot_idx += 1

        # Plot each model's colorized depth
        for model in sets:
            ax = plt.subplot(total_rows, total_cols, subplot_idx)
            ax.imshow(plot_per_model[model]["depth_colored"])
            ax.set_title(f"{model} Map", fontsize=9)
            ax.axis("off")
            subplot_idx += 1

        # Plot each model's overlay
        for model in sets:
            ax = plt.subplot(total_rows, total_cols, subplot_idx)
            ax.imshow(plot_per_model[model]["overlay"])
            ax.set_title(f"{model} Overlay (A={alpha})", fontsize=9)
            ax.axis("off")
            subplot_idx += 1

    # Adjust subplot spacing to minimize white space
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=wspace, hspace=hspace)
    
    # Add comment if provided
    if comment:
        fig.suptitle(comment, fontsize=10, y=0.995)
        plt.subplots_adjust(top=0.98)

    if output_dir:
        if rgb_path_for_save:
            out_file = get_save_path(rgb_path_for_save, output_dir)
        else:
            # Fallback: construct path directly if no RGB path available
            out_file = os.path.join(os.path.abspath(output_dir), "comparison_plot_compact.png")
        out_file = get_unique_path(Path(out_file))
        logger.info("Figure saved to:", extra={"path": str(out_file)})
        plt.savefig(out_file, bbox_inches='tight', pad_inches=pad_inches, dpi=150)
    else:
        plt.show()
        logger.info("Displayed comparison plot - not saved")
    plt.close()

