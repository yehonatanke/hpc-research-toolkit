import os
from PIL import Image
from collections import Counter
import json
from typing import Dict, Any

def analyze_image_resolutions(data_dir: str, output_file: str, extensions: tuple = ('.jpg', '.jpeg', '.png')) -> None:    
    print(f"\nStarting scan in folder: {data_dir}...")
    
    resolution_counts: Counter = Counter() # Counts unique resolution
    all_image_data: Dict[str, str] = {}    # list: [file name: resolution]
    stats_data: Dict[str, Any] = {}
    
    # Counting and iterating through files
    total_images_scanned = 0
    
    # os.walk recursively 
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_path = os.path.join(root, file)
                total_images_scanned += 1
                
                try:
                    with Image.open(image_path) as img:
                        # Get resolution 
                        width, height = img.size
                        resolution_key = f"{width}x{height}"
                        
                        resolution_counts[resolution_key] += 1
                        all_image_data[image_path] = resolution_key
                        
                except Exception as e:
                    print(f"Error processing file {image_path}: {e}")
                    all_image_data[image_path] = f"ERROR: {e}"

    # Creating the summary of the statistics
    stats_data['total_images_scanned'] = total_images_scanned
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
        
    stats_data['all_image_resolutions'] = all_image_data

    # Saving the data to a JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=4, ensure_ascii=False)
        
        print(f"\nAnalysis completed successfully.")
        print(f"Total images scanned: {total_images_scanned}")
        print(f"Most common resolution: {stats_data['most_common_resolution'].get('resolution', 'N/A')}")
        print(f"The complete statistics are saved in: {output_file}")
        
    except Exception as e:
        print(f"Error saving output file: {e}")

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