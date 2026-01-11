import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


def _convert_extrinsic_to_homogeneous(extrinsic: np.ndarray) -> np.ndarray:
    """
    Convert 3x4 extrinsic matrix to 4x4 homogeneous matrix.
    Args:
        extrinsic: 3x4 extrinsic matrix (or 4x4)
    Returns:
        4x4 homogeneous matrix with bottom row [0, 0, 0, 1]
    """
    logger.debug(f"Converting extrinsic matrix to homogeneous: {extrinsic.shape}")
    if extrinsic.shape == (3, 4):
        homogeneous = np.eye(4, dtype=np.float32)
        homogeneous[:3, :] = extrinsic  # Copy [R|t] to top 3 rows
    elif extrinsic.shape == (4, 4):
        homogeneous = extrinsic
    else:
        raise ValueError(f"Unexpected extrinsic shape: {extrinsic.shape}, expected (3, 4) or (4, 4)")
    return homogeneous


def load_camera_pose_from_npz(cam_file: Dict) -> np.ndarray:
    """
    Load camera pose/extrinsic matrix from npz file and convert to 4x4 homogeneous format.

    Args:
        cam_file: Dictionary-like object from np.load() containing camera parameters

    Returns:
        np.ndarray: Camera pose matrix as 4x4 homogeneous transformation matrix (float32)
            Shape: (4, 4)
            Format: [R | t] where R is 3x3 rotation, t is 3x1 translation
                    [0 | 1] bottom row is [0, 0, 0, 1]

    Raises:
        KeyError: If neither "extrinsic" nor "pose" key exists in cam_file
        ValueError: If the matrix shape is not (3, 4) or (4, 4)
    """
    logger.debug(f"Loading camera pose from npz file: {cam_file.keys()}")
    # try "extrinsic" first
    if "extrinsic" in cam_file:
        logger.debug(f"Loading extrinsic matrix from npz file: {cam_file['extrinsic'].shape}")
        extrinsic = cam_file["extrinsic"].astype(np.float32)
        camera_pose = _convert_extrinsic_to_homogeneous(extrinsic)
    elif "pose" in cam_file:
        logger.warning(
            f"Loading pose matrix from npz file: {cam_file['pose'].shape}, but for RE10K, we use extrinsic matrix"
        )
        camera_pose = cam_file["pose"].astype(np.float32)
        camera_pose = _convert_extrinsic_to_homogeneous(camera_pose)
    else:
        logger.error(
            f"Camera file must contain either 'extrinsic' or 'pose' key. Available keys: {list(cam_file.keys())}"
        )
        raise KeyError(
            f"Camera file must contain either 'extrinsic' or 'pose' key. Available keys: {list(cam_file.keys())}"
        )

    return camera_pose
