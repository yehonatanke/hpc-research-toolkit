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

    # config_path = args.config_path or DEFAULT_CONFIG
    config_path = args.config_path if args.config_path else DEFAULT_CONFIG
    config = load_config(config_path=config_path)

    logger.debug(f"Config path: {os.path.abspath(config_path)}")
    if args.config_path:
        logger.debug("Custom config path:", extra={"path": os.path.abspath(args.config_path)})
    else:
        logger.debug("Default Config path:", extra={"path": os.path.abspath(DEFAULT_CONFIG)})

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--export-dir", type=str, default=config.get("export_dir"))
    parser.add_argument("--dense_dataset", action="store_true")
    parser.add_argument("--alpha", type=float, default=config.get("alpha", 0.6))
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--log_path", type=str, default=config.get("log_path"))
    parser.add_argument("--config_path", type=str, default=config_path)
    parser.add_argument("--comment", type=str, default="")

    log_path = config.get("log_path")
    if log_path:
        logger.debug("Log path from config:", extra={"path": log_path})
        logger.debug("Absolute log path:", extra={"path": os.path.abspath(log_path)})

    return parser.parse_args()
