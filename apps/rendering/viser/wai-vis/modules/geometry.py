import numpy as np
import cv2
from typing import Dict, Tuple, List
from .logger import debug_print

def parse_transform_matrix(transform: List) -> np.ndarray:
    return np.array(transform).reshape(4, 4)

def calculate_quaternion_from_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to quaternion [w, x, y, z]."""
    try:
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(rotation_matrix)
        wxyz = rotation.as_quat()  # Returns [x, y, z, w]
        return np.array([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])
    except ImportError:
        debug_print(1, "[Warning] Scipy not found. Using simplified manual quaternion conversion.")
        # Simplified manual fallback
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (rotation_matrix[2,1] - rotation_matrix[1,2]) / s
            y = (rotation_matrix[0,2] - rotation_matrix[2,0]) / s
            z = (rotation_matrix[1,0] - rotation_matrix[0,1]) / s
            return np.array([w, x, y, z])
        return np.array([1.0, 0.0, 0.0, 0.0])

def unproject_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Dict[str, float],
    transform_matrix: np.ndarray,
    depth_scale: float = 1.0,
    max_depth: float = 1000.0,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """Unproject depth map to 3D points using pinhole camera model."""
    
    depth_h, depth_w = depth.shape
    rgb_h, rgb_w = rgb.shape[:2]
    
    # Resize depth if dimensions don't match
    if depth_h != rgb_h or depth_w != rgb_w:
        depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
    
    # downsample (slicing)
    if downsample > 1:
        depth = depth[::downsample, ::downsample]
        rgb = rgb[::downsample, ::downsample]

    h, w = depth.shape
    
    # Scale intrinsics
    # if downsample_intrinsics :
    fl_x = intrinsics.get('fl_x', intrinsics.get('fx', 0)) / downsample
    fl_y = intrinsics.get('fl_y', intrinsics.get('fy', 0)) / downsample
    cx = intrinsics.get('cx', w / 2.0) / downsample
    cy = intrinsics.get('cy', h / 2.0) / downsample
    
    # Create coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Mask valid depth
    depth_scaled = depth * depth_scale
    valid_mask = (depth_scaled > 0) & (depth_scaled < max_depth)
    
    # Unproject (pinhole camera model)
    x_cam = (u - cx) * depth_scaled / fl_x
    y_cam = (v - cy) * depth_scaled / fl_y
    z_cam = depth_scaled
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_cam = points_cam[valid_mask]
    
    # handle colors
    if rgb.ndim == 3:
        colors = rgb[valid_mask]
    else:
        colors = np.stack([rgb[valid_mask]]*3, axis=-1)
    
    # transform to world (camera-to-world)
    if len(points_cam) > 0:
        points_cam_hom = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
        points_world = (transform_matrix @ points_cam_hom.T).T[:, :3]
        return points_world, colors
    
    return np.array([]), np.array([])

