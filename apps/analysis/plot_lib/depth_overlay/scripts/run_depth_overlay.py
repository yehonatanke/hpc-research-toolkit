import os
import logging

from src.cli import parse_args
from research_utils import plot_overlay, print_args


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    print_args(args)
    logger.debug("Script started with args", extra={"input_args": vars(args)})

    plot_overlay(rgb_path=args.rgb_path, depth_path=args.depth_path, save_dir=args.export_dir, model_name=args.model_name, alpha=args.alpha)

    logger.debug("Export full path:", extra={"path": f"{os.path.abspath(args.export_dir)}"})
    logger.debug("RGB Full path:", extra={"path": f"{os.path.abspath(args.rgb_path)}"})
    logger.debug("-->Process finished<--")


if __name__ == "__main__":
    main()
