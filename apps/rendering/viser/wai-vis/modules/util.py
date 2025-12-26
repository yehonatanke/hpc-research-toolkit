import argparse
import numpy as np
from typing import Dict

def get_da3_metric_depth(depth: np.ndarray, intrinsics: Dict[str, float]) -> np.ndarray:
    """
    docs: https://github.com/ByteDance-Seed/Depth-Anything-3/tree/main?tab=readme-ov-file#-faq
    """
    focal_avg = (intrinsics['fl_x'] + intrinsics['fl_y']) / 2.0
    depth = (focal_avg * depth) / 300.0
    return depth


def print_args_to_log(args, log_file):
    print(f"\n===== ARGUMENTS =====", file=log_file)
    for k, v in vars(args).items():
        print(f"\t-{k}: {v}", file=log_file)
    print(f"===== END OF ARGUMENTS =====\n", file=log_file)


def print_args_to_log_v1(args, log_file):
    print(f"\n===========[INFO] ARGS:===========", file=log_file)
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}", file=log_file)
    print(f"===================================\n", file=log_file)

def print_args(args):
    print(f"\n===========[INFO] ARGS:===========")
    for arg in vars(args):
        print(f"- {arg}: {getattr(args, arg)}")
    print(f"===================================\n")

def print_args_v1(args):
    print(f"\n==========================================")
    print(f"[INFO] ARGS:")
    print(f"- root: {args.dataset_root}")
    print(f"- port: {args.port}")
    print(f"- host: {args.host}")
    print(f"- frame_skip: {args.frame_skip}")
    print(f"- downsample: {args.downsample}")
    print(f"- default_depth_scale: {args.default_depth_scale}")
    print(f"- default_max_depth: {args.default_max_depth}")
    print(f"- default_point_size: {args.default_point_size}")
    print(f"- dataset: {args.dataset}")
    print(f"- depth_source: {args.depth_source}")
    print(f"- debug: {args.debug}")
    print(f"- log_filepath: {args.log_filepath}")
    print(f"==========================================\n")