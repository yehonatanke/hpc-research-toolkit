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

    logger.info(f"Config path: {os.path.abspath(config_path)}")
    if args.config_path:
        logger.info(f"Custom config path: {os.path.abspath(args.config_path)}")
    else:
        logger.info(f"Default Config path: {os.path.abspath(DEFAULT_CONFIG)}")

    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_path", type=str, required=True)
    parser.add_argument("--depth_path", type=str, required=True)
    parser.add_argument("--export-dir", type=str, default=config.get("export_dir"))
    parser.add_argument("--dense_dataset", action="store_true")
    parser.add_argument("--alpha", type=float, default=config.get("alpha", 0.6))
    parser.add_argument("--log_path", type=str, default=config.get("log_path"))
    parser.add_argument("--config_path", type=str, default=config_path)

    # Debug info (only print if log_path exists)
    log_path = config.get("log_path")
    if log_path:
        logger.debug(f"Log path from config: {log_path}")
        logger.debug(f"Absolute log path: {os.path.abspath(log_path)}")
    runloger()
    return parser.parse_args()

def runloger():
    logger = logging.getLogger(__name__)

    logger.debug("Debug message", extra={"request_id": "abc123"})
    logger.info("Application started")
    logger.warning("Something suspicious")
    logger.error("Failed to connect", exc_info=True)
    # Option 1: Use % formatting (recommended for single path)
    logger.info("Loading model from: %s", "./models/best.pth")

    # Option 2: Use .extra={} for structured extras
    logger.info("Loading model from:", extra={"path": "./models/best.pth"})

    # Option 3: Multiple extras
    logger.info("Processing file:", extra={"path": "/data/train.csv", "request_id": "abc123"})
    exit()
# def parse_args():
#     # Preliminary parse to check for a custom config path
#     conf_parser = argparse.ArgumentParser(add_help=False)
#     conf_parser.add_argument("--config_path", type=str)
#     args, _ = conf_parser.parse_known_args()

#     # Load specified or default configuration
#     config_path = args.config_path if args.config_path else "default.json"
#     print(f"Config path: {config_path}")
#     if args.config_path:
#         print(f"custom config path: {os.path.abspath(args.config_path)}")
#     else:
#         print(f"default Config path: {os.path.abspath('configs/default.json')}")
#     config = load_config(config_filename=config_path)

#     # Main parser with dynamic defaults from the loaded config
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rgb_path", type=str, required=True)
#     parser.add_argument("--depth_path", type=str, required=True)
#     parser.add_argument("--export-dir", type=str, default=config.get("export_dir"))
#     parser.add_argument("--dense_dataset", action="store_true")
#     parser.add_argument("--alpha", type=float, default=config.get("alpha", 0.6))
#     parser.add_argument("--log_path", type=str, default=config.get("log_path"))
#     parser.add_argument("--config_path", type=str, default=config_path)

#     return parser.parse_args()

# def parse_args():
#     config_defaults = load_config(config_filename="default.json")

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--rgb_path", type=str, required=True)
#     parser.add_argument("--depth_path", type=str, required=True)
#     parser.add_argument("--export-dir", type=str, default=config_defaults["export_dir"])
#     parser.add_argument("--dense_dataset", action="store_true", help="Use dense dataset")
#     parser.add_argument("--alpha", type=float, default=0.6)
#     parser.add_argument("--log_path", type=str, default=config_defaults["log_path"])
#     parser.add_argument("--config_path", type=str, default=config_defaults["config_path"])

#     return parser.parse_args()
