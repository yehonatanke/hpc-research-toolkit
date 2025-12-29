# from modules.cli import get_parser
# from modules.logger import setup_logging, close_logging, set_debug_level, debug_print
# from modules.util import print_args
# from modules.plot import plot_comparison_multiple


import os
import logging

from src.cli import parse_args
from research_utils import plot_overlay, print_args, print_args_color


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    print_args_color(args)
    logger.debug("Script started with args", extra={"input_args": vars(args)}) 

    try:
        plot_comparison_multiple(
                scene_root=args.scene_root,
                depths_to_compare=args.depths_to_compare, # dense
                dense_dir=args.dense_dir, # 
                output_dir=args.export_dir,
                alpha=args.alpha,
                skip_step=args.skip_step,
                width_ratio=args.width_ratio,
                height_ratio=args.height_ratio,
                comment=args.comment, 
            )

    except Exception as e:
        logger.error(f"Error: {e}")
    finally: # edit args here and up
        logger.debug("Export full path:", extra={"path": f"{os.path.abspath(args.export_dir)}"})
        logger.debug("RGB Full path:", extra={"path": f"{os.path.abspath(args.rgb_path)}"})
        logger.debug("-->Process finished<--")


if __name__ == "__main__":
    main()

