#!/usr/bin/env python3

import argparse
from pathlib import Path
from modules.logger import setup_logging, close_logging, set_debug_level, debug_print
from modules.dataset import WAI_Dataset, DenseDataset
from modules.renderer import ViserApp
from modules.util import print_args


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True, help="Path to dataset root directory")
    parser.add_argument("--port", type=int, default=8080, help="Port for Viser server (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for Viser server (default: 0.0.0.0)")
    parser.add_argument("--frame_skip", type=int, default=1, help="Process every Nth frame (default: 1, process all)")
    parser.add_argument("--downsample", type=int, default=4, help="Downsample factor for point cloud (default: 4)")
    parser.add_argument("--default_depth_scale", type=float, default=1.0, help="Default depth scale (default: 1.0)")
    parser.add_argument("--default_max_depth", type=float, default=100.0, help="Default max depth cutoff (default: 100.0)")
    parser.add_argument("--default_point_size", type=float, default=0.01, help="Default point size (default: 0.01)")
    parser.add_argument("--dataset", type=str, default="mvsanywhere", choices=["mvsanywhere", "moge"],
                        help="Dataset type to use: 'mvsanywhere' or 'moge' (default: mvsanywhere)")
    parser.add_argument("--depth_source", type=str, default=None, choices=["mvsanywhere_depth", "moge2_depth"],
                        help="Depth source key to use from scene_meta.json. If not provided, will be inferred from --dataset")
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1, 2, 3], help="Debug level (default: 0)")
    parser.add_argument("--log_filepath", type=str, default=None, help="Path to log file (default: None)")
    parser.add_argument("--dense_dataset", action="store_true", help="Use dense dataset")
    parser.add_argument("--da3_scaling", action="store_true", help="Use DA3 scaling factor for depth")
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Setup Logging
    setup_logging(args.log_filepath, args)
    set_debug_level(args.debug)

    print_args(args)

    # --- [!] IF ADDING NEW DATASET, EDIT HERE [!] ---
    try:
        # --- [!] Dense dataset [!] ---
        if args.dense_dataset:
            dataset = DenseDataset(
                    root=Path(args.dataset_root),
                    frame_skip=args.frame_skip,
                    da3_scaling=args.da3_scaling
                )
        # --- [!] WAI dataset [!] ---
        else:
            # Determine Depth Key
            if args.depth_source:
                depth_key = args.depth_source
            else:
                depth_key = "moge2_depth" if args.dataset == "moge" else "mvsanywhere_depth"
            
            debug_print(1, f"Initializing Dataset: {args.dataset_root} with key: {depth_key}")
            # Initialize Dataset
            dataset = WAI_Dataset(
                root=Path(args.dataset_root),
                frame_skip=args.frame_skip,
                depth_source_key=depth_key
            )
        
        # Initialize & Run App
        app_defaults = {
            'depth_scale': args.default_depth_scale,
            'max_depth': args.default_max_depth,
            'point_size': args.default_point_size
        }
        
        app = ViserApp(
            dataset=dataset,
            host=args.host,
            port=args.port,
            downsample=args.downsample,
            defaults=app_defaults
        )
        
        app.run()

    except Exception as e:
        debug_print(0, f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        close_logging()


if __name__ == "__main__":
    main()