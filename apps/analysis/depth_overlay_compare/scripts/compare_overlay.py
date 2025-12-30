# from modules.cli import get_parser
# from modules.logger import setup_logging, close_logging, set_debug_level, debug_print
# from modules.util import print_args
# from modules.plot import plot_comparison_multiple


import os
import sys
import logging

# Add project root to path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add research_utils to path (it's in the parent code directory)
code_root = os.path.dirname(project_root)
research_utils_path = os.path.join(code_root, "research_utils", "src")
if research_utils_path not in sys.path:
    sys.path.insert(0, research_utils_path)

from src.cli import parse_args
from research_utils import print_args_color, plot_comparison_multiple, plot_comparison_multiple_compact


logger = logging.getLogger(__name__)


def main():
    args = parse_args()
    print_args_color(args)
    logger.debug("Script started with args", extra={"input_args": vars(args)}) 

    plot_comparison_multiple_compact(
            scene_root=args.scene_root,
            models=args.models, 
            dense_dir=args.dense_dir, 
            output_dir=args.export_dir,
            alpha=args.alpha,
            skip_step=args.skip_step,
            width_ratio=args.width_ratio,
            height_ratio=args.height_ratio,
            comment=args.comment, 
        )
 
    logger.debug("Export full path:", extra={"path": f"{os.path.abspath(args.export_dir)}"})
    logger.debug("-->Process finished<--")


if __name__ == "__main__":
    main()

