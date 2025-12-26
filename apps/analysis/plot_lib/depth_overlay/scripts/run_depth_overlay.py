import os
import logging
import cv2

from src.cli import parse_args
from research_utils import plot_overlay, print_args


logger = logging.getLogger(__name__)

def main():
    args = parse_args()
    print_args(args)
    logger.info("Script started with args", extra={"input_args": vars(args)})

    plot_overlay(rgb_path=args.rgb_path, depth_path=args.depth_path, save_dir=args.export_dir, alpha=args.alpha)

    # print full path of the export directory
    # print(f"Full path: {os.path.abspath(args.export_dir)}")
    # print(f"Full path: {os.path.abspath(args.export_dir)}")
    logger.info(f"- Export Full path: {os.path.abspath(args.export_dir)}")
    logger.info(f"- RGB Full path: {os.path.abspath(args.rgb_path)}")
    logger.info("Process finished")

if __name__ == "__main__":
    main()
