from modules.cli import get_parser
from modules.logger import setup_logging, close_logging, set_debug_level, debug_print
from modules.util import print_args
from modules.plot import plot_comparison_multiple


def main():
    parser = get_parser()
    args = parser.parse_args()

    # setup logging
    setup_logging(args.log_dir, args)
    set_debug_level(args.debug)

    print_args(args)

    try:
        plot_comparison_multiple(
                scene_root=args.scene_root,
                depths_to_compare=args.depths_to_compare,
                output_dir=args.output_dir,
                alpha=args.alpha,
                skip_step=args.skip_step,
                width_ratio=args.width_ratio,
                height_ratio=args.height_ratio,
                save=args.save
            )
    except Exception as e:
        debug_print(1, f"Error: {e}")
        raise e
    finally:
        close_logging()


if __name__ == "__main__":
    main()

