import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from research_utils.utils.io import ensure_dir, get_unique_path, get_save_path
from research_utils.utils.log import logger
from research_utils.utils.data import extract_wai_meta_paths, extract_dense_paths, subsample_paths
from research_utils.utils.plot import get_plot_dimensions, process_depth
from research_utils.utils.data import extract_wai_meta_paths, extract_dense_paths, subsample_paths

# problems with depth
def plot_depth_comparison_grid(
    sets: dict,
    models: list,
    num_frames: int,
    alpha: float,
    output_dir: str,
    scene_root: str,
    comment: str
):
    num_models = len(models)
    fig, axes = plt.subplots(num_frames, num_models + 1, figsize=(4 * (num_models + 1), 4 * num_frames))
    
    if num_frames == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(num_frames):
        first_model = models[0]
        rgb_path = sets[first_model]['rgb_paths'][i]
        
        if not os.path.exists(rgb_path):
            continue
            
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Frame {i}\nOriginal RGB")
        axes[i, 0].axis('off')

        for j, model in enumerate(models):
            depth_path = sets[model]['depth_paths'][i]
            
            if not os.path.exists(depth_path):
                axes[i, j + 1].text(0.5, 0.5, f"Missing:\n{os.path.basename(depth_path)}", 
                                    ha='center', va='center')
                axes[i, j + 1].axis('off')
                continue

            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
                
            depth = depth.astype(np.float32)
            
            # Normalization
            d_min, d_max = depth.min(), depth.max()
            depth_norm = (depth - d_min) / (d_max - d_min + 1e-6)
            
            # Colormap
            depth_color = plt.cm.viridis(depth_norm)[:, :, :3]
            depth_color = (depth_color * 255).astype(np.uint8)
            
            if depth_color.shape[:2] != rgb.shape[:2]:
                depth_color = cv2.resize(depth_color, (rgb.shape[1], rgb.shape[0]))

            overlay = cv2.addWeighted(rgb, 1 - alpha, depth_color, alpha, 0)
            
            axes[i, j + 1].imshow(overlay)
            axes[i, j + 1].set_title(f"{model}")
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    if comment:
        plt.suptitle(comment, fontsize=16)
        plt.subplots_adjust(top=0.95)
        
    scene_name = os.path.basename(scene_root.rstrip('/'))
    save_path = os.path.join(output_dir, f"comparison_{scene_name}.png")
    save_path = get_unique_path(Path(save_path))
    logger.info("Figure saved to:", extra={"path": str(save_path)})
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
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

    # Get actual image dimensions for resolution preservation
    img_height_px = None
    img_width_px = None
    if preserve_resolution:
        # Load first image to get dimensions
        first_model = next(iter(sets.keys()))
        first_rgb_path = sets[first_model]["rgb_paths"][0]
        first_img = cv2.imread(first_rgb_path, cv2.IMREAD_COLOR)
        if first_img is not None:
            img_height_px, img_width_px = first_img.shape[:2]
            logger.info("Image dimensions detected:", extra={"width": img_width_px, "height": img_height_px})

    total_cols, fig_width, fig_height = get_plot_dimensions(models, num_frames, width_ratio, height_ratio)
    
    # If preserving resolution, calculate figure size to maintain 1:1 pixel mapping
    if preserve_resolution and img_width_px and img_height_px:
        # Calculate figure size based on image dimensions and DPI
        # Each subplot should be img_width_px/dpi inches wide to maintain pixel-perfect display
        subplot_width_inches = img_width_px / dpi
        subplot_height_inches = img_height_px / dpi
        
        # Total figure size = subplot size * number of subplots
        # wspace and hspace are fractions, so we add minimal spacing
        # Account for spacing between subplots (wspace is fraction of subplot width)
        spacing_width = wspace * subplot_width_inches * (total_cols - 1) if total_cols > 1 else 0
        spacing_height = hspace * subplot_height_inches * (num_frames - 1) if num_frames > 1 else 0
        
        fig_width = subplot_width_inches * total_cols + spacing_width
        fig_height = subplot_height_inches * num_frames + spacing_height
        
        logger.info("Figure size calculated from image resolution:", 
                   extra={"fig_width": fig_width, "fig_height": fig_height, "dpi": dpi,
                          "subplot_size": f"{subplot_width_inches:.2f}x{subplot_height_inches:.2f} inches"})

    logger.info("Plotting dimensions:", extra={"total_cols": total_cols, "fig_width": fig_width, "fig_height": fig_height})
    logger.info(f"Plotting {num_frames} frames on a {fig_width}x{fig_height} grid with minimal spacing (DPI: {dpi}).")
    
    # Create figure and axes grid using subplots for better control
    fig, axes = plt.subplots(num_frames, total_cols, figsize=(fig_width, fig_height), dpi=dpi)
    
    # Handle single row case
    if num_frames == 1:
        axes = axes.reshape(1, -1)
    
    total_rows = num_frames

    for i in tqdm(range(num_frames)):
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

        # Plot RGB image
        col_idx = 0
        ax = axes[i, col_idx]
        ax.imshow(rgb_img)
        if show_titles:
            ax.set_title(f"{frame_title} - RGB", fontsize=title_fontsize, pad=2)
        ax.axis("off")
        ax.set_aspect('auto')
        col_idx += 1

        # Plot each model's colorized depth
        for model in sets:
            ax = axes[i, col_idx]
            ax.imshow(plot_per_model[model]["depth_colored"])
            if show_titles:
                ax.set_title(f"{model} Map", fontsize=title_fontsize, pad=2)
            ax.axis("off")
            ax.set_aspect('auto')
            col_idx += 1

        # Plot each model's overlay
        for model in sets:
            ax = axes[i, col_idx]
            ax.imshow(plot_per_model[model]["overlay"])
            if show_titles:
                ax.set_title(f"{model} Overlay (A={alpha})", fontsize=title_fontsize, pad=2)
            ax.axis("off")
            ax.set_aspect('auto')
            col_idx += 1

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