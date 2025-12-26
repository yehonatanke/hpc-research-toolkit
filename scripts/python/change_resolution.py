import os
import json
import shutil
from PIL import Image 
import argparse
import sys
from typing import Any

def update_scene_metadata(root_directory: str):
    """
    Iterates through subfolders, checks image resolution, and updates 
    the corresponding json metadata to match.
    """
    
    for subdir in os.listdir(root_directory):
        subdir_path = os.path.join(root_directory, subdir)
        
        if not os.path.isdir(subdir_path):
            continue

        images_path = os.path.join(subdir_path, "images_distorted")
        json_path = os.path.join(subdir_path, "scene_meta_distorted.json")

        # Check if both required components exist
        if not os.path.exists(images_path) or not os.path.exists(json_path):
            print(f"Skipping {subdir}: Missing images or json file.")
            continue

        # Get image resolution
        try:
            # Find the first image
            first_image_path = None
            with os.scandir(images_path) as entries:
                for entry in entries:
                    if entry.is_file() and entry.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        first_image_path = entry.path
                        break
            
            if not first_image_path:
                print(f"Skipping {subdir}: No images found in folder.")
                continue                
            
            with Image.open(first_image_path) as img:
                real_w, real_h = img.size
                
        except Exception as e:
            print(f"Error reading image in {subdir}: {e}")
            continue

        # JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Get dimensions
            meta_w = data.get('w')
            meta_h = data.get('h')

            if meta_w == real_w and meta_h == real_h:
                print(f"Skipping {subdir}: Resolution matches ({real_w}x{real_h}).")
                continue

            print(f"Updating {subdir}...")
            print(f"  - Changing from [{meta_w}x{meta_h}] to [{real_w}x{real_h}]")

            # scaling factors
            scale_x = real_w / meta_w
            scale_y = real_h / meta_h

            # create backup
            # shutil.copy(json_path, json_path + ".bak")

            data['w'] = real_w
            data['h'] = real_h

            if 'fl_x' in data: data['fl_x'] *= scale_x
            if 'fl_y' in data: data['fl_y'] *= scale_y
            if 'cx' in data:   data['cx']   *= scale_x
            if 'cy' in data:   data['cy']   *= scale_y

            # updated 
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            print(f"[SUCCESS] Updated {subdir}.")

        except Exception as e:
            print(f"[ERROR] Failed to update {subdir} metadata: {e}")

    return True

def parse_args() -> Any:
    parser = argparse.ArgumentParser(description="Change resolution of images and update scene metadata")
    parser.add_argument("--root", type=str, required=True, help="Root directory containing scene folders")
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    success = update_scene_metadata(root_directory=args.root)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())