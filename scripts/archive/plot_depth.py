import os 
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1" # enable openexr support
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

def plot_exr_depth(file_path, save_path):

    depth_map = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    if depth_map is None:
        print(f"Error: Could not read file {file_path}")
        return

    if len(depth_map.shape) > 2:
        depth_map = depth_map[:, :, 0]

    depth_map = np.nan_to_num(depth_map, posinf=0, neginf=0)

    print(f"Stats for {file_path}:")
    print(f"  Min Depth: {np.min(depth_map):.4f}")
    print(f"  Max Depth: {np.max(depth_map):.4f}")
    print(f"  Mean Depth: {np.mean(depth_map):.4f}")

    plt.figure(figsize=(10, 6))
    
    plt.imshow(depth_map, cmap='plasma') 
    plt.colorbar(label='Depth Value')
    plt.title(f'Depth Map: {file_path.split("/")[-1]}')
    plt.axis('off')
    plt.savefig(os.path.join(save_path, f'depth_map_01.png')) 
    plt.close()
    print(f"Saved depth map to {os.path.join(save_path, f'depth_map_01.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True, default='/leonardo_work/AIFAC_S02_060/data/yk/debug/output/depth_maps')
    args = parser.parse_args()

    plot_exr_depth(args.file_path, args.save_path)