import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from ..io.image import get_rgb_image, get_depth_map, get_image_dimensions
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


def plot_overlay_one(rgb_path, depth_path, output_dir, alpha=0.6):
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        logger.error(f"Error reading RGB: {rgb_path}")
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  

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
        logger.warning(
            f"Resizing depth from {depth_colored.shape} to {rgb_img.shape}",
            extra={"rgb_path": rgb_path, "depth_path": depth_path},
        )
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
