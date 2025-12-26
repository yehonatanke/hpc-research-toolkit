import viser
import numpy as np
import imageio.v3 as iio
import json
import argparse
import os
from pathlib import Path


def resolve_path(json_path, relative_path):
    json_dir = Path(json_path).parent
    return json_dir / relative_path


def load_scene_metadata(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Scene metadata file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        meta = json.load(f)
    
    # Validate required fields
    required_fields = ['fl_x', 'fl_y', 'cx', 'cy', 'w', 'h', 'frames']
    for field in required_fields:
        if field not in meta:
            raise ValueError(f"Missing required field in metadata: {field}")
    
    if not meta['frames']:
        raise ValueError("No frames found in metadata")
    
    return meta


def load_frame_data(json_path, frame, meta):
    """Load image, depth, and normals for a frame."""
    # Resolve paths relative to JSON file location
    image_path = resolve_path(json_path, frame['file_path'])
    depth_path = resolve_path(json_path, frame.get('moge2_depth', ''))
    normals_path = resolve_path(json_path, frame.get('moge2_normals', ''))
    
    # Load RGB image
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    rgb = iio.imread(str(image_path))
    
    # Load depth (EXR format)
    depth = None
    if depth_path.exists():
        depth = iio.imread(str(depth_path)).astype(np.float32)
    elif 'mvsanywhere_depth' in frame:
        # Fallback to mvsanywhere depth if moge2_depth not available
        depth_path = resolve_path(json_path, frame['mvsanywhere_depth'])
        if depth_path.exists():
            depth = iio.imread(str(depth_path)).astype(np.float32)
    
    if depth is None:
        raise FileNotFoundError(f"Depth file not found for frame: {frame.get('frame_name', 'unknown')}")
    
    # Load normals (EXR format)
    normals = None
    if normals_path.exists():
        normals = iio.imread(str(normals_path)).astype(np.float32)
    
    return rgb, depth, normals


def backproject_depth(depth, fx, fy, cx, cy):
    """Back-project depth map to 3D points in camera space."""
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into (H, W, 3) array
    points_camera = np.stack((x, y, z), axis=-1)
    return points_camera


def transform_points_to_world(points_camera, transform_matrix):
    """Transform points from camera space to world space using transform_matrix."""
    # Convert to homogeneous coordinates
    H, W, _ = points_camera.shape
    points_flat = points_camera.reshape(-1, 3)
    ones = np.ones((points_flat.shape[0], 1))
    points_homogeneous = np.hstack([points_flat, ones])
    
    # Apply transform matrix (4x4)
    transform = np.array(transform_matrix)
    points_world_homogeneous = (transform @ points_homogeneous.T).T
    points_world = points_world_homogeneous[:, :3]
    
    return points_world.reshape(H, W, 3)


def transform_normals_to_world(normals_camera, transform_matrix):
    """Transform normals from camera space to world space."""
    transform = np.array(transform_matrix)
    rotation = transform[:3, :3]
    
    H, W, _ = normals_camera.shape
    normals_flat = normals_camera.reshape(-1, 3)
    normals_world_flat = (rotation @ normals_flat.T).T
    normals_world = normals_world_flat.reshape(H, W, 3)
    
    # Normalize to ensure unit length
    norm = np.linalg.norm(normals_world, axis=-1, keepdims=True)
    normals_world = np.where(norm > 0, normals_world / norm, normals_world)
    
    return normals_world


def visualize_with_intrinsics(json_path, frame_index=0):
    """Visualize a scene frame with point cloud and normals in Viser."""
    # Load scene metadata
    meta = load_scene_metadata(json_path)
    
    # Validate frame index
    if frame_index < 0 or frame_index >= len(meta['frames']):
        raise ValueError(f"Frame index {frame_index} out of range [0, {len(meta['frames'])-1}]")
    
    frame = meta['frames'][frame_index]
    frame_name = frame.get('frame_name', f'frame_{frame_index:05d}')
    
    # Get camera intrinsics
    fx = meta['fl_x']
    fy = meta['fl_y']
    cx = meta['cx']
    cy = meta['cy']
    W = meta['w']
    H = meta['h']
    
    rgb, depth, normals = load_frame_data(json_path, frame, meta)
    
    if rgb.shape[:2] != (H, W):
        print(f"Warning: Image dimensions {rgb.shape[:2]} don't match metadata ({H}, {W})")
        print("Attempting to resize image...")
        try:
            from PIL import Image
            rgb = np.array(Image.fromarray(rgb).resize((W, H)))
        except ImportError:
            print("PIL not available. Cropping/padding image to match metadata dimensions.")
            h_actual, w_actual = rgb.shape[:2]
            if h_actual >= H and w_actual >= W:
                # Crop center
                start_h = (h_actual - H) // 2
                start_w = (w_actual - W) // 2
                rgb = rgb[start_h:start_h+H, start_w:start_w+W]
            else:
                # Pad with zeros
                rgb_padded = np.zeros((H, W) + rgb.shape[2:], dtype=rgb.dtype)
                min_h = min(h_actual, H)
                min_w = min(w_actual, W)
                rgb_padded[:min_h, :min_w] = rgb[:min_h, :min_w]
                rgb = rgb_padded
    
    if depth.shape != (H, W):
        raise ValueError(f"Depth map dimensions {depth.shape} don't match metadata ({H}, {W})")
    
    points_camera = backproject_depth(depth, fx, fy, cx, cy)
    
    if 'transform_matrix' in frame:
        points_world = transform_points_to_world(points_camera, frame['transform_matrix'])
    else:
        print("Warning: No transform_matrix found, using camera-space coordinates")
        points_world = points_camera
    
    points = points_world.reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    
    server = viser.ViserServer()
    
    camera_convention = meta.get('camera_convention', 'opencv')
    if camera_convention == 'opencv':
        server.scene.set_up_direction("+z")
    
    server.scene.add_point_cloud(
        name=f"/scene/{frame_name}/points",
        points=points,
        colors=colors,
        point_size=0.01,
        point_shape='circle'
    )
    
    # Add normals visualization if available
    if normals is not None:
        if normals.shape != (H, W, 3):
            print(f"Warning: Normals dimensions {normals.shape} don't match expected ({H}, {W}, 3)")
        else:
            # Transform normals to world space
            if 'transform_matrix' in frame:
                normals_world = transform_normals_to_world(normals, frame['transform_matrix'])
            else:
                normals_world = normals
            
            stride = 15
            p_sampled = points_world[::stride, ::stride, :].reshape(-1, 3)
            n_sampled = normals_world[::stride, ::stride, :].reshape(-1, 3)
            
            line_ends = p_sampled + n_sampled * 0.05
            lines = np.stack((p_sampled, line_ends), axis=1)
            
            server.scene.add_line_segments(
                name=f"/scene/{frame_name}/normals",
                points=lines,
                colors=(0, 255, 0),  # green
                line_width=1.0
            )
    
    scene_name = meta.get('scene_name', 'unknown')
    print(f"Scene: {scene_name}")
    print(f"Frame: {frame_name} (index {frame_index})")
    print(f"Viser server is up at {server.get_host()}:{server.get_port()}")
    server.sleep_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize DL3DV scene frames with Viser")
    parser.add_argument("--json", required=True, help="Path to scene_meta.json")
    parser.add_argument("--index", type=int, default=0, help="Frame index to visualize (default: 0)")
    args = parser.parse_args()

    visualize_with_intrinsics(args.json, args.index)


