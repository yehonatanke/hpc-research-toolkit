import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"  # enable openexr support

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_overlay(rgb_path, depth_path, save_path, alpha=0.6):
    # load rgb image
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        print(f"Error reading RGB: {rgb_path}")
        return
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)  # convert to matplotlib format

    # load depth map
    depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_map is None:
        print(f"Error reading Depth: {depth_path}")
        return

    # handle extra channels if any
    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    # replace NaN/Inf with 0
    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

    # normalize depth map to 0-255 range (required to blend with regular image)
    # we use Min-Max normalization for visualization
    d_min, d_max = depth_map.min(), depth_map.max()
    depth_normalized = (depth_map - d_min) / (d_max - d_min + 1e-8)  # avoid division by 0
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)

    # colorize depth map
    depth_colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_PLASMA)
    depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)

    # check if shapes match (sometimes the output is smaller than the original)
    if rgb_img.shape != depth_colored.shape:
        print(f"Resizing depth from {depth_colored.shape} to {rgb_img.shape}")
        depth_colored = cv2.resize(depth_colored, (rgb_img.shape[1], rgb_img.shape[0]))

    # blend of two images
    overlay = cv2.addWeighted(rgb_img, 1 - alpha, depth_colored, alpha, 0)

    # plot: original, depth, and overlay
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

    # extract filename from path
    # filename = os.path.basename(rgb_path)
    filename = os.path.basename(os.path.dirname(os.path.dirname(rgb_path))) + "_" + os.path.basename(rgb_path)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.close()
    print(f"Saved overlay to {os.path.join(save_path, filename)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True, default="$DEBUG/output/depth_overlay")
    parser.add_argument("--dense_dataset", action="store_true", help="Use dense dataset")
    args = parser.parse_args()

    plot_overlay(args.rgb_path, args.depth_path, args.save_path)
