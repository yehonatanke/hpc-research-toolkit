import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def get_rgb_image(rgb_path: str):
    img = cv2.imread(rgb_path)
    if img is None:
        logger.error("RGB file not found", extra={"path": rgb_path})
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_depth_map(depth_path: str):
    if depth_path.endswith(".npy"):
        depth_map = np.load(depth_path)
        logger.debug("Depth file is a numpy array", extra={"path": depth_path})
    else:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        logger.debug("Depth file is not a numpy array", extra={"path": depth_path})

    if depth_map is None:
        logger.error("Depth file not found", extra={"path": depth_path})
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    return depth_map


def get_save_path(rgb_path, save_dir):
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
    filename = f"{parent_folder}/{os.path.basename(rgb_path)}"
    abs_path = os.path.abspath(save_dir)
    full_path = os.path.join(abs_path, filename)

    logger.debug("Generating save path:")
    logger.debug("full_path", extra={"path": full_path})
    logger.debug("abs_path", extra={"path": abs_path})
    logger.debug("file_name", extra={"path": filename})

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def plot_overlay(rgb_path, depth_path, save_dir, model_name, alpha=0.6):
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

    if model_name:
        plt.gcf().text(
            0.02,
            0.96,
            f"Model: {model_name}",
            fontsize=10,
            color="blue",
            horizontalalignment="left",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round"),
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
