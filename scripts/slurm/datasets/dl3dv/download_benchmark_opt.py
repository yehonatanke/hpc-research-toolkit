import os
from os.path import join
import pandas as pd
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download
import argparse
import pickle
import shutil
import concurrent.futures
import time

# FORCE RUST DOWNLOADER (If installed)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

api = HfApi()
repo_root = "DL3DV/DL3DV-10K-Benchmark"


def download_single_file(repo_path: str, odir: str):
    rel_path = os.path.relpath(repo_path, repo_root)
    cache_dir = join(odir, ".cache")

    # Aggressive retry for HPC (5 retries, small backoff)
    for attempt in range(5):
        try:
            hf_hub_download(
                repo_id=repo_root, filename=rel_path, repo_type="dataset", local_dir=odir, cache_dir=cache_dir
            )
            return True
        except Exception as e:
            # If 429 (Too Many Requests), wait a bit
            if "429" in str(e):
                time.sleep(1 + attempt)
            continue
    return False


def clean_huggingface_cache(cache_dir: str):
    target = join(cache_dir, "datasets--DL3DV--DL3DV-10K-Benchmark")
    if os.path.exists(target):
        shutil.rmtree(target)


def download_by_hash(filepath_dict: dict, odir: str, hash_code: str, only_level4: bool, workers: int):
    if hash_code not in filepath_dict:
        return False

    all_files = filepath_dict[hash_code]
    files_to_download = []

    for f in all_files:
        if only_level4:
            if ("images" in f and "images_4" not in f) or "input" in f:
                continue
        files_to_download.append(join(repo_root, f))

    if not files_to_download:
        return True

    # Use the user-defined worker count
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(download_single_file, f, odir) for f in files_to_download]
        for future in concurrent.futures.as_completed(futures):
            if not future.result():
                return False

    return True


def download_benchmark(args):
    output_dir = args.odir
    subset_opt = args.subset

    os.makedirs(output_dir, exist_ok=True)

    # 1. Metadata (Single thread is fine here)
    print("Fetching metadata...")
    if not download_single_file(join(repo_root, "benchmark-meta.csv"), output_dir):
        return False
    if not download_single_file(join(repo_root, ".cache/filelist.bin"), output_dir):
        return False

    df = pd.read_csv(join(output_dir, "benchmark-meta.csv"))
    filepath_dict = pickle.load(open(join(output_dir, ".cache/filelist.bin"), "rb"))
    download_list = [args.hash] if subset_opt == "hash" else df["hash"].tolist()

    print(f"Downloading with {args.workers} threads...")
    for cur_hash in tqdm(download_list):
        if not download_by_hash(filepath_dict, output_dir, cur_hash, args.only_level4, args.workers):
            return False
        if args.clean_cache:
            clean_huggingface_cache(join(output_dir, ".cache"))

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--odir", type=str, default="DL3DV-10K-Benchmark")
    parser.add_argument("--subset", choices=["full", "hash"], required=True)
    parser.add_argument("--only_level4", action="store_true")
    parser.add_argument("--clean_cache", action="store_true")
    parser.add_argument("--hash", type=str, default="")
    # allows to push to 128 via CLI
    parser.add_argument("--workers", type=int, default=64, help="Number of parallel downloads")
    params = parser.parse_args()

    download_benchmark(params)
