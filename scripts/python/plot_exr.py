import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


def visualize_exr_normals(args):
    normals = cv2.imread(args.file_path, cv2.IMREAD_UNCHANGED)

    if normals is None:
        print(f"[ERROR] Could not read file: {args.file_path}")
        return

    normals_rgb = cv2.cvtColor(normals, cv2.COLOR_BGR2RGB)

    visual_normals = (normals_rgb + 1) / 2.0

    visual_normals = np.clip(visual_normals, 0, 1)

    plt.figure(figsize=(10, 5))
    plt.imshow(visual_normals)
    plt.title(f"Surface Normals: {os.path.basename(args.file_path)}")
    plt.axis("off")
    if args.save_path is not None:
        filename = (
            os.path.basename(os.path.dirname(os.path.dirname(args.file_path))) + "_" + os.path.basename(args.file_path)
        )
        filename = filename.replace(".exr", ".png")
        plt.savefig(os.path.join(args.save_path, filename))
        print(f"[INFO] Saved figure to: {os.path.join(args.save_path, filename)}")
    else:
        plt.show()
        print(f"[INFO] Showing figure...")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True, help="path to the exr file")
    parser.add_argument("--save_path", type=str, default=None, help="if None, only plot to screen")
    return parser


def main(args):
    visualize_exr_normals(args)


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
