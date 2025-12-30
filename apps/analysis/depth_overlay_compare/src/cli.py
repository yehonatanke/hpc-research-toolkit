#!/usr/bin/env python3

import argparse
from research_utils import load_config
import os
import logging

from research_utils import setup_logging

DEFAULT_CONFIG = "configs/default.json"
DEFAULT_LOGGING_CONFIG = "configs/default_logging.json"


def parse_args():
    conf_parser = argparse.ArgumentParser(add_help=False)
    conf_parser.add_argument("--config_path", type=str)
    conf_parser.add_argument("--log_config", type=str, default=DEFAULT_LOGGING_CONFIG)
    args, _ = conf_parser.parse_known_args()

    setup_logging(args.log_config)
    logger = logging.getLogger(__name__)

    config_path = args.config_path if args.config_path else DEFAULT_CONFIG
    config = load_config(config_path=config_path)

    logger.debug(f"Config path: {os.path.abspath(config_path)}")
    if args.config_path:
        logger.debug("Custom config path:", extra={"path": os.path.abspath(args.config_path)})
    else:
        logger.debug("Default Config path:", extra={"path": os.path.abspath(DEFAULT_CONFIG)})

    # main parser
    parser = argparse.ArgumentParser(
        description=(
            "Compare depth overlays of multiple models.\n"
            "NOTE: Unless otherwise specified, default values for arguments are read from the default config file (%s)."
            % config_path
        )
    )
    parser.add_argument("--scene_root", type=str, required=True, help="Path to scene root directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=config.get("models"),
        choices=["mvsanywhere", "moge2", "depthanything"],
        help="Models to compare: 'mvsanywhere' | 'moge2' | 'depthanything'",
    )
    parser.add_argument(
        "--dense_dir", type=str, help="Path to dense directory. must be specified if 'dense' is in --depths_to_compare"
    )  # take care
    parser.add_argument("--skip_step", type=int, default=config.get("skip_step"), help="Number of frames to skip")
    parser.add_argument(
        "--width_ratio", type=float, default=config.get("width_ratio"), help="The width ratio of the figure"
    )
    parser.add_argument(
        "--height_ratio", type=float, default=config.get("height_ratio"), help="The height ratio of the figure"
    )
    parser.add_argument("--alpha", type=float, default=config.get("alpha"), help="The alpha value for the overlay")
    parser.add_argument(
        "--export_dir",
        type=str,
        default=config.get("export_dir"),
        help="Path to output directory",
    )
    parser.add_argument("--log_path", type=str, default=config.get("log_path"))
    parser.add_argument("--comment", type=str, default="", help="Comment for the output image file")

    parser.epilog = "Default values for most arguments are loaded from the config file unless specified otherwise."

    log_path = config.get("log_path")
    if log_path:
        logger.debug("Log path from config:", extra={"path": log_path})
        logger.debug("Absolute log path:", extra={"path": os.path.abspath(log_path)})

    return parser.parse_args()


def get_parserOLD():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", type=str, required=True, help="Path to scene root directory")
    parser.add_argument(
        "--depths_to_compare",
        type=list,
        default=["mvsanywhere_depth", "moge2_depth"],
        choices=["mvsanywhere_depth", "moge2_depth"],
        help="Depth sources to compare: 'mvsanywhere_depth' or 'moge2_depth' (default: ['mvsanywhere_depth', 'moge2_depth'])",
    )
    parser.add_argument("--skip_step", type=int, default=0, help="Number of frames to skip (default: 0)")
    parser.add_argument("--width_ratio", type=float, default=4, help="Width ratio of the figure (default: 4)")
    parser.add_argument("--height_ratio", type=float, default=4, help="Height ratio of the figure (default: 4)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha value for the overlay (default: 0.6)")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/leonardo_work/AIFAC_S02_060/data/yk/code/depth-overlay-compare/output",
        help="Path to output directory",
    )
    parser.add_argument(
        "--save", action="store_true", default=False, help="If True, save output images (default: False)"
    )
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1, 2, 3], help="Debug level (default: 0)")
    parser.add_argument(
        "--log_dir",
        type=str,
        default="/leonardo_work/AIFAC_S02_060/data/yk/code/depth-overlay-compare/logs",
        help="Path to log directory (default: /depth-overlay-compare/logs)",
    )

    return parser
