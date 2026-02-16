
TARGET="${CODE}/scripts/slurm/datasets/dl3dv/benckmark/hashes/hashes.txt"

# 1. Generate hashes.txt 
python -c "import pandas as pd; from huggingface_hub import hf_hub_download; f = hf_hub_download(repo_id='DL3DV/DL3DV-10K-Benchmark', filename='benchmark-meta.csv', repo_type='dataset'); print('\n'.join(pd.read_csv(f)['hash'].tolist()))"  > ${TARGET}

# 2. Split into 10 files (chunk_00 to chunk_09)
split -d -n l/10 ${TARGET} chunk_


### RUN IN SHELL TO PRINT ONLY ###

### Count Top-Level Items
# python3 -c "from huggingface_hub import HfFileSystem; fs = HfFileSystem(); print(len(fs.ls('datasets/DL3DV/DL3DV-10K-Benchmark', detail=False)))"

### Exclude Metadata/Scripts (Count only Scene Folders - excluding `README.md`, `download.py`, etc.)
# python3 -c "from huggingface_hub import HfFileSystem; fs = HfFileSystem(); files = fs.ls('datasets/DL3DV/DL3DV-10K-Benchmark', detail=False); print(len([f for f in files if len(f.split('/')[-1]) == 64]))"
