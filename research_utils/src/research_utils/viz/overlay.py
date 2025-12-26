import os
import cv2
import numpy as np
import matplotlib.pyplot as plt  
import logging

logger = logging.getLogger(__name__)


def get_save_path(rgb_path, save_dir):
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
    filename = f"{parent_folder}/{os.path.basename(rgb_path)}"
    abs_path = os.path.abspath(save_dir)
    full_path = os.path.join(abs_path, filename)

    logger.info("Generating save path", extra={"full_path": full_path, "abs_path": abs_path, "file_name": filename})

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def plot_overlay(rgb_path, depth_path, save_dir, alpha=0.6):
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        logger.error("RGB file not found", extra={"path": rgb_path})
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)

    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        logger.error("Depth file not found", extra={"path": depth_path})
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_normalized = (depth_map - d_min) / (d_max - d_min + 1e-8)
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    if rgb_img.shape[:2] != depth_colored.shape[:2]:
        logger.warning(
            "Dimension mismatch, resizing depth map",
            extra={"rgb_shape": rgb_img.shape[:2], "depth_shape": depth_colored.shape[:2]},
        )
        depth_colored = cv2.resize(depth_colored, (rgb_img.shape[1], rgb_img.shape[0]))

    overlay = cv2.addWeighted(rgb_img, 1 - alpha, depth_colored, alpha, 0)

    plt.figure(figsize=(15, 5))
    images = [rgb_img, depth_colored, overlay]
    titles = ["RGB", "Depth", f"Overlay ({alpha})"]

    for i, (img, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, 3, i)
        plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    out_file = get_save_path(rgb_path, save_dir)

    logger.info(f"Saving figure to {out_file}")
    plt.savefig(out_file)
    plt.close()
    return out_file
