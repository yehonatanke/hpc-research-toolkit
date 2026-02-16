import os
import argparse
from huggingface_hub import snapshot_download, hf_hub_download
import time

# --- CONFIGURATION ---
# DISABLE Turbo mode (It causes 429 errors on small files)
if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
    del os.environ["HF_HUB_ENABLE_HF_TRANSFER"]

REPO_ID = "DL3DV/DL3DV-10K-Benchmark"


def download_benchmark(args):
    odir = args.odir

    print(f"--- SAFE DOWNLOAD MODE ---")
    print(f"Target: {odir}")
    print(f"Threads: 4 (Throttled to prevent 429 bans)")
    print(f"Strategy: Single Snapshot (No loops)")

    # Define filters to only get what you need
    allow_patterns = []

    if args.only_level4:
        # Get all images_4 folders and all colmap data
        allow_patterns.append("**/images_4/**")
        allow_patterns.append("**/colmap/**")
        # Also get the essential metadata files
        allow_patterns.append("benchmark-meta.csv")
        allow_patterns.append(".cache/filelist.bin")
    else:
        allow_patterns = None  # Download everything

    # If user wants a specific scene, refine the pattern
    if args.subset == "hash":
        if not args.hash:
            print("Error: --hash is required.")
            return
        # Restrict to just that folder
        if allow_patterns:
            allow_patterns = [f"{args.hash}/{p}" for p in allow_patterns]
        else:
            allow_patterns = [f"{args.hash}/**"]

    # THE EXECUTION
    # We use a while loop to handle the 429s gracefully if they still happen
    max_retries = 10
    for attempt in range(max_retries):
        try:
            snapshot_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                local_dir=odir,
                allow_patterns=allow_patterns,
                max_workers=4,
            )
            print("\nSUCCESS: Download Complete!")
            break
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg:
                wait_time = 60 * (attempt + 1)
                print(f"\n[429 Hit] Too fast. Cooling down for {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"\nError: {e}")
                # If it's not a rate limit, it might be a real network error, wait briefly and retry
                time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--odir", type=str, default="DL3DV-10K-Benchmark")
    parser.add_argument("--subset", choices=["full", "hash"], required=True)
    parser.add_argument("--only_level4", action="store_true")
    parser.add_argument("--hash", type=str, default="")

    args = parser.parse_args()
    download_benchmark(args)
