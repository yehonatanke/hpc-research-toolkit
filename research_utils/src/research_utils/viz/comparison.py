import os
import cv2
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm
from typing import List
from pathlib import Path

from ..io.image import get_image_dimensions
from ..core.path import get_save_path, ensure_dir, extract_wai_meta_paths, extract_dense_paths, get_unique_path
from ..core.util import subsample_paths
from .comparison_util import (
    process_depth,
    _build_model_sets,
    _calculate_figure_size,
    _process_frame_data,
    _plot_frame_row,
    _apply_figure_layout,
    _save_or_show_figure,
    get_plot_dimensions,
)

logger = logging.getLogger(__name__)


# first version
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
        elif model == "depthanything":
            result = extract_dense_paths(dense_dir, model)
            result = subsample_paths(result, skip_step)
            if result and result.get("rgb_paths"):
                sets["depthanything"] = result

    non_empty_lengths = {k: len(v.get("rgb_paths", [])) for k, v in sets.items() if v and v.get("rgb_paths")}
    if not non_empty_lengths:
        logger.error("No valid depth maps found for any model in {models}")
        return

    # any available integer value (e.g., the first)
    num_frames = next(iter(non_empty_lengths.values()), 0)

    total_cols, fig_width, fig_height = get_plot_dimensions(models, num_frames, width_ratio, height_ratio)

    logger.info(
        "Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height}
    )
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid.")
    plt.figure(figsize=(fig_width, fig_height))

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

    sets, rgb_path_for_save = _build_model_sets(scene_root, models, dense_dir, skip_step)

    non_empty_lengths = {k: len(v.get("rgb_paths", [])) for k, v in sets.items() if v and v.get("rgb_paths")}
    if not non_empty_lengths:
        logger.error("No valid depth maps found for any model in {models}")
        return

    num_frames = next(iter(non_empty_lengths.values()), 0)

    img_width_px, img_height_px = None, None
    if preserve_resolution and sets:
        first_model = next(iter(sets.keys()))
        first_rgb_path = sets[first_model]["rgb_paths"][0]
        img_width_px, img_height_px = get_image_dimensions(first_rgb_path)
        if img_width_px and img_height_px:
            logger.info("Image dimensions detected:", extra={"width": img_width_px, "height": img_height_px})

    total_cols, fig_width, fig_height = _calculate_figure_size(
        models,
        num_frames,
        width_ratio,
        height_ratio,
        preserve_resolution,
        img_width_px,
        img_height_px,
        dpi,
        wspace,
        hspace,
    )

    logger.info(
        "Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height}
    )
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid with minimal spacing (DPI: {dpi}).")

    fig, axes = plt.subplots(num_frames, total_cols, figsize=(fig_width, fig_height), dpi=dpi)

    if num_frames == 1:
        axes = axes.reshape(1, -1)

    for i in tqdm(range(num_frames)):
        rgb_img, frame_title, plot_per_model = _process_frame_data(sets, i, alpha)
        _plot_frame_row(axes[i], rgb_img, frame_title, plot_per_model, sets, alpha, show_titles, title_fontsize)

    _apply_figure_layout(fig, show_titles, wspace, hspace, comment)

    _save_or_show_figure(fig, output_dir, rgb_path_for_save, dpi, pad_inches)


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

    non_empty_lengths = {k: len(v.get("rgb_paths", [])) for k, v in sets.items() if v and v.get("rgb_paths")}
    if not non_empty_lengths:
        logger.error("No valid depth maps found for any model in {models}")
        return

    # any available integer value (e.g., the first)
    num_frames = next(iter(non_empty_lengths.values()), 0)

    total_cols, fig_width, fig_height = get_plot_dimensions(models, num_frames, width_ratio, height_ratio)

    logger.info(
        "Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height}
    )
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid with compact spacing.")

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

    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=wspace, hspace=hspace)

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
        plt.savefig(out_file, bbox_inches="tight", pad_inches=pad_inches, dpi=150)
    else:
        plt.show()
        logger.info("Displayed comparison plot - not saved")
    plt.close()
