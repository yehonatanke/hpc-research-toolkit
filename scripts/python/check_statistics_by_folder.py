import os
from PIL import Image
from collections import Counter
import json
from typing import Dict, Any, List

def analyze_image_resolutions_by_folder(data_dir: str, output_file: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> bool:    
    print(f"\nStarting multi-folder scan in root: {data_dir}...")
    
    # Define the specific folders to look for: 1K through 11K
    target_folders = [f"{i}K" for i in range(1, 12)]
    
    # hold the report for all folders
    # Structure: { "1K": { stats... }, "2K": { stats... } }
    global_report: Dict[str, Any] = {} # key: folder name, value: folder stats
    
    for folder_name in target_folders:
        full_folder_path = os.path.join(data_dir, folder_name)
        
        # Initialize stats for this specific folder
        folder_stats: Dict[str, Any] = {}
        resolution_counts: Counter = Counter()
        file_data: Dict[str, str] = {}
        total_images_scanned = 0
        
        print(f"- Processing folder: {folder_name}")
        
        if not os.path.exists(full_folder_path):
            print(f"    [!] Folder {folder_name} not found. Skipping.")
            global_report[folder_name] = {"status": "Folder not found"}
            continue

        for root, _, files in os.walk(full_folder_path):
            for file in files:
                if file.lower().endswith(extensions):
                    image_path = os.path.join(root, file)
                    total_images_scanned += 1
                    
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            resolution_key = f"{width}x{height}"
                            
                            resolution_counts[resolution_key] += 1
                            file_data[file] = resolution_key # Saving filename 
                            
                    except Exception as e:
                        print(f"    Error processing file {file}: {e}")
                        file_data[file] = f"ERROR: {e}"

        folder_stats['status'] = "Scanned"
        folder_stats['total_images'] = total_images_scanned
        folder_stats['unique_resolutions_count'] = len(resolution_counts)
        folder_stats['resolution_frequency'] = dict(resolution_counts.most_common())
        
        if resolution_counts:
            most_common = resolution_counts.most_common(1)[0]
            folder_stats['most_common_resolution'] = {
                "resolution": most_common[0],
                "count": most_common[1]
            }
        else:
            folder_stats['most_common_resolution'] = "No images found"
            
        folder_stats['file_details'] = file_data
        
        # Add this folder's stats to the global report
        global_report[folder_name] = folder_stats

        print(f"    Completed analysis for: {folder_name}")

    # --- Save Global Report ---
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(global_report, f, indent=4, ensure_ascii=False)
        
        print(f"\nAnalysis completed.")
        print(f"Report saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='The root directory containing 1K, 2K... folders', required=True)
    parser.add_argument('--output_file', type=str, help='The JSON file to save the results', required=True)
    args = parser.parse_args()
    
    if analyze_image_resolutions_by_folder(args.data_dir, args.output_file):
        print(f"[FUNCTION:analyze_image_resolutions_by_folder] [STATE:success]")
    else:
        print(f"[FUNCTION:analyze_image_resolutions_by_folder] [STATE:failed]")