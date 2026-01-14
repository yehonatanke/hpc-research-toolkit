import os
import logging
import json
from pathlib import Path


logger = logging.getLogger(__name__)


def sys_full_path(env_var, sub_path):
    """
    Input: sys_full_path($WORK, "data/")
    Output: /home/user/work/data/
    """
    base = os.environ.get(env_var, "")
    logger.debug("BASE", extra={"path": base})
    logger.debug("RETURN PATH", extra={"path": os.path.join(base, sub_path)})
    # print(f"BASE: {base}")
    # print(f"RETURN PATH: {os.path.join(base, sub_path)}")
    return os.path.join(base, sub_path)


def extract_dense_paths(dense_dir: str, model: str = None) -> dict:
    """
    Extracts paths from dense directory
    Args:
        - dense_dir: path to dense directory
        - model: model name
    Returns:
        - dict with keys: 'rgb_paths', 'frame_names', 'depth_paths'
    """
    if model is None:
        logger.warning("Function got no model to extract")
        return None

    # dense_dir/dense/rgb or dense_dir/rgb
    rgb_path = _get_dense_path(dense_dir, "rgb")
    depth_path = _get_dense_path(dense_dir, "depth")

    result = {"rgb_paths": [], "frame_names": [], "depth_paths": []}

    # get `/dense_dir/depth/frame_00001.npy` etc.
    # Sort files to ensure consistent ordering (matching scene_meta.json order)
    files = sorted(os.listdir(rgb_path))
    for file in files:
        frame_name = file.split(".")[0]
        result["frame_names"].append(frame_name)
        result["rgb_paths"].append(os.path.join(rgb_path, file))
        # Depth files are .npy, not .png
        depth_file = f"{frame_name}.npy"
        result["depth_paths"].append(os.path.join(depth_path, depth_file))

    logger.info(f"model: {model} --> extracted {len(result['depth_paths'])} depth paths")
    return result


def _get_dense_path(dense_dir: str, sub_dir: str) -> str:
    """
    Adds "dense" suffix to the path
    Args:
        - path: path to the file
    Returns:
        - path with "dense" suffix
    Note: This is due to dense_dummy and dense test variations
    """
    if os.path.exists(os.path.join(dense_dir, sub_dir)):
        # <dense_dir>/<sub_dir>
        return os.path.join(dense_dir, sub_dir)
    # <dense_dir>/dense/<sub_dir>
    return os.path.join(dense_dir, "dense", sub_dir)


def get_abs_path(root: str, path: str) -> str:
    return os.path.join(root, path)


def extract_wai_meta_paths(scene_root: str, model_json_key: str = None) -> dict:
    """
    Extracts paths from wai's scene_meta.json
    Args:
        - scene_root: root directory of the scene
        - model_json_key: key of the model in the scene_meta.json
    Returns:
        - dict with keys: 'rgb_paths', 'frame_names', 'depth_paths'
    """
    if model_json_key is None:
        logger.warning("Function got no model_json_key to extract")
        return None

    wai_meta_path = os.path.join(scene_root, "scene_meta.json")
    result = {"rgb_paths": [], "frame_names": [], "depth_paths": []}

    with open(wai_meta_path, "r") as f:
        data = json.load(f)

    for frame in data.get("frames", []):
        result["rgb_paths"].append(get_abs_path(scene_root, frame.get("image")))
        result["frame_names"].append(frame.get("frame_name"))
        depth = frame.get(model_json_key)
        if depth:
            result["depth_paths"].append(get_abs_path(scene_root, depth))

    logger.info(f"model key: {model_json_key} --> extracted {len(result['rgb_paths'])} rgb paths")
    logger.info(f"model key: {model_json_key} --> extracted {len(result['depth_paths'])} depth paths")

    return result


def extract_meta_paths(scene_meta_path, models=["mvsanywhere", "moge2", "mapanything"]):
    """
    Extracts depths from scene_meta.json
    Args:
        - scene_meta_path: path to scene_meta.json
        - depths_to_compare: list of depths to compare
    Returns:
        - rgb_paths: list of rgb paths
        - frame_names: list of frame names
        - depth_paths_1: list of depth paths for the first depth model
        - depth_paths_2: list of depth paths for the second depth model
    """
    if len(models) < 2:
        logger.warning("Function got less than 2 models to compare")

    rgb_paths = []
    frame_names = []
    depth_paths_1 = []
    depth_paths_2 = []

    with open(scene_meta_path, "r") as f:
        data = json.load(f)

    if "mapanything" not in models:
        for frame in data.get("frames", []):
            rgb_paths.append(frame.get("image"))
            frame_names.append(frame.get("frame_name"))
            depth_1 = frame.get(models[0])
            depth_2 = frame.get(models[1])

            if depth_1:
                depth_paths_1.append(depth_1)
            if depth_2:
                depth_paths_2.append(depth_2)

    logger.info(f"Extracted {len(rgb_paths)} rgb paths")
    logger.info(f"Extracted {len(depth_paths_1)} depth paths for {models[0]}")
    logger.info(f"Extracted {len(depth_paths_2)} depth paths for {models[1]}")

    return rgb_paths, frame_names, depth_paths_1, depth_paths_2


def get_unique_path(filepath: Path) -> Path:
    """
    Returns a unique Path object. If the original path exists,
    it appends a counter (e.g., _1, _2) before the file extension.
    """
    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent

    counter = 1
    while True:
        new_filename = f"{stem}_{counter}{suffix}"
        new_filepath = parent / new_filename

        if not new_filepath.exists():
            return new_filepath

        counter += 1


def ensure_dir(dir_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        - dir_path: Path to the directory to ensure exists
    """
    if not dir_path:
        return

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        logger.debug("Created directory", extra={"path": dir_path})
    else:
        logger.debug("Directory already exists", extra={"path": dir_path})


def get_save_path(rgb_path, save_dir):
    parent_folder = os.path.basename(os.path.dirname(os.path.dirname(rgb_path)))
    filename = f"{parent_folder}/{os.path.basename(rgb_path)}"
    abs_path = os.path.abspath(save_dir)
    full_path = os.path.join(abs_path, filename)

    logger.debug("Generating save path:")
    logger.debug("full_path", extra={"path": full_path})
    logger.debug("abs_path", extra={"path": abs_path})
    logger.debug("file_name", extra={"path": filename})

    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path


def path_with_parents(pathname: str, num_levels: int = 2) -> str:
    """
    Returns 'parent2/parent1/filename' (num_levels parents before file) for the given path.

    Args:
        pathname: The full path to process
        num_levels: Number of parent directories to include (default: 2)

    Returns:
        A relative path string with the specified number of parent directories and the filename.
        Returns empty string if pathname is empty.

    Examples:
        >>> path_with_parents("/a/b/c/d/file.py", 2)
        'c/d/file.py'
        >>> path_with_parents("/a/b/file.py", 2)
        'b/file.py'
        >>> path_with_parents("file.py", 2)
        'file.py'
    """
    if not pathname:
        return ""
    parts = os.path.normpath(pathname).split(os.sep)
    if len(parts) >= (num_levels + 1):
        rel_path = os.path.join(*parts[-(num_levels + 1) :])
    elif len(parts) >= 2:
        rel_path = os.path.join(parts[-2], parts[-1])
    else:
        rel_path = parts[-1]
    return rel_path
