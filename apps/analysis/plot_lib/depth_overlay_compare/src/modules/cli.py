#!/usr/bin/env python3

import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", type=str, required=True, help="Path to scene root directory")
    parser.add_argument("--depths_to_compare", type=list, default=["mvsanywhere_depth", "moge2_depth"], choices=["mvsanywhere_depth", "moge2_depth"],
                        help="Depth sources to compare: 'mvsanywhere_depth' or 'moge2_depth' (default: ['mvsanywhere_depth', 'moge2_depth'])")
    parser.add_argument("--skip_step", type=int, default=0, help="Number of frames to skip (default: 0)")
    parser.add_argument("--width_ratio", type=float, default=4, help="Width ratio of the figure (default: 4)")
    parser.add_argument("--height_ratio", type=float, default=4, help="Height ratio of the figure (default: 4)")
    parser.add_argument("--alpha", type=float, default=0.6, help="Alpha value for the overlay (default: 0.6)")
    parser.add_argument("--output_dir", type=str, default='/leonardo_work/AIFAC_S02_060/data/yk/code/depth-overlay-compare/output', help="Path to output directory")
    parser.add_argument("--save", action='store_true', default=False, help="If True, save output images (default: False)")
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1, 2, 3], help="Debug level (default: 0)")
    parser.add_argument("--log_dir", type=str, default="/leonardo_work/AIFAC_S02_060/data/yk/code/depth-overlay-compare/logs", help="Path to log directory (default: /depth-overlay-compare/logs)")
    
    return parser


