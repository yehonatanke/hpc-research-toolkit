import os
from PIL import Image
from collections import Counter
import json
from typing import Dict, Any

def analyze_image_resolutions(data_dir: str, output_file: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> bool:    
    print(f"\nStarting scan in folder: {data_dir}...")
    
    # Target resolution (exclude from logging)
    TARGET_W, TARGET_H = 480, 270
    
    resolution_counts: Counter = Counter() 
    file_data: Dict[str, str] = {}    # hold non-480x270 images
    stats_data: Dict[str, Any] = {}
    
    total_images_scanned = 0
    non_standard_count = 0
    
    # os.walk recursively 
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_path = os.path.join(root, file)
                total_images_scanned += 1
                
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                        resolution_key = f"{width}x{height}"
                        resolution_counts[resolution_key] += 1
                        
                        # --- LOGGING LOGIC ---
                        # Only add to file list if it is NOT 480x270
                        if width != TARGET_W or height != TARGET_H:
                            file_data[file] = resolution_key
                            non_standard_count += 1
                        
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")
                    file_data[file] = f"ERROR: {e}"

    # summary 
    stats_data['total_images_scanned'] = total_images_scanned
    stats_data['images_logged_(non_480x270)'] = non_standard_count
    stats_data['unique_resolutions_count'] = len(resolution_counts)
    stats_data['resolution_frequency'] = dict(resolution_counts.most_common())
    
    # Finding the most common resolution
    if resolution_counts:
        most_common_res = resolution_counts.most_common(1)[0]
        stats_data['most_common_resolution'] = {
            "resolution": most_common_res[0],
            "count": most_common_res[1]
        }
    else:
        stats_data['most_common_resolution'] = "No images found."
        
    # contains only the non-480x270
    stats_data['file_details'] = file_data

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        
        print(f"\nAnalysis completed successfully.")
        print(f"Total images scanned: {total_images_scanned}")
        print(f"Non-standard images found: {non_standard_count}")
        print(f"Most common resolution: {stats_data['most_common_resolution'].get('resolution', 'N/A')}")
        print(f"The complete statistics are saved in: {output_file}")
        return True
        
    except Exception as e:
        print(f"Error saving output file: {e}")
        return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='The directory to scan for image resolutions', required=True)
    parser.add_argument('--output_file', type=str, help='The file to save the results', required=True)
    args = parser.parse_args()
    
    if analyze_image_resolutions(args.data_dir, args.output_file):
        print(f"[FUNCTION:analyze_image_resolutions] [STATE:success] --- SAVE DIRECTORY:{args.output_file}")
    else:
        print(f"[FUNCTION:analyze_image_resolutions] [STATE:failed] --- SAVE DIRECTORY:{args.output_file}")