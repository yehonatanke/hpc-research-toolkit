#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import viser
import inspect
import sys


LOG_FILE_HANDLER = None
DEBUG_LEVEL = 0


def setup_logging(log_filepath="debug.log", args=None):
    """Sets up the global file handler for debug_print."""
    global LOG_FILE_HANDLER

    if not log_filepath:
        print(f"\n[INFO] No log file path provided. Debug messages will print to console (sys.stderr).\n")
        return
    try:
        LOG_FILE_HANDLER = open(log_filepath, 'w')
        print(f"\n[INFO] Messages will be logged to: {os.path.abspath(log_filepath)}\n")
        if args:
            print(f"\n===== ARGUMENTS =====", file=LOG_FILE_HANDLER)
            print(f"\t-log_filepath: {os.path.abspath(log_filepath)}", file=LOG_FILE_HANDLER)
            print(f"\t-debug: {args.debug}", file=LOG_FILE_HANDLER)
            print(f"\t-dataset_root: {args.dataset_root}", file=LOG_FILE_HANDLER)
            print(f"\t-port: {args.port}", file=LOG_FILE_HANDLER)
            print(f"\t-host: {args.host}", file=LOG_FILE_HANDLER)
            print(f"\t-frame_skip: {args.frame_skip}", file=LOG_FILE_HANDLER)
            print(f"\t-downsample: {args.downsample}", file=LOG_FILE_HANDLER)
            print(f"\t-default_depth_scale: {args.default_depth_scale}", file=LOG_FILE_HANDLER)
            print(f"\t-default_max_depth: {args.default_max_depth}", file=LOG_FILE_HANDLER)
            print(f"\t-dataset: {args.dataset}", file=LOG_FILE_HANDLER)
            print(f"\t-depth_source: {args.depth_source}", file=LOG_FILE_HANDLER)
            print(f"===== END OF ARGUMENTS =====\n", file=LOG_FILE_HANDLER)
    except IOError as e:
        print(f"[Error] Could not open log file {log_filepath}: {e}", file=sys.stderr)
        LOG_FILE_HANDLER = sys.stderr


def close_logging():
    """Closes the global log file handler if it was opened."""
    global LOG_FILE_HANDLER
    if LOG_FILE_HANDLER and LOG_FILE_HANDLER != sys.stdout:
        LOG_FILE_HANDLER.close()
        LOG_FILE_HANDLER = None


def debug_print(level: int, *args, **kwargs):
    """
    Print debug message only if current DEBUG_LEVEL >= level
    """
    output_stream = LOG_FILE_HANDLER if LOG_FILE_HANDLER else sys.stderr
    if DEBUG_LEVEL >= level:
        # Get the frame (the function)
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name
        
        # Get the file name and line number
        full_path = frame.f_code.co_filename
        filename = os.path.basename(full_path)
        lineno   = frame.f_lineno
        prefix = f"[DEBUG] [{func_name}() @ {filename}:{lineno}]"
        
        print(prefix, *args, **kwargs, file=output_stream)

        if output_stream == sys.stderr:
            output_stream.flush()

# ==========================================
# Data Loading Utilities
# ==========================================

def load_depth_map(depth_path: Path) -> Optional[np.ndarray]:
    """Load depth map from various formats (.exr, .png, .npy)."""
    if not depth_path.exists():
        return None
    
    ext = depth_path.suffix.lower()
    
    if ext == '.exr':
        try:
            import OpenEXR
            import Imath
            exr_file = OpenEXR.InputFile(str(depth_path))
            header = exr_file.header()
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            for channel_name in ['R', 'Y', 'Z', 'G']:
                try:
                    depth_str = exr_file.channel(channel_name, Imath.PixelType(Imath.PixelType.FLOAT))
                    depth = np.frombuffer(depth_str, dtype=np.float32)
                    depth = depth.reshape((height, width))
                    debug_print(1, f"Loaded depth map (from EXR file) with: depth.shape=[{depth.shape}].")
                    return depth
                except:
                    continue
            debug_print(1, f"No channel worked, returning None.")
            return None
        except ImportError:
            debug_print(1, f"ImportError: Failed to import OpenEXR library.")
            pass
        
        # OpenCV Fallback
        try:
            import imageio.v2 as imageio
            depth = imageio.imread(str(depth_path))
            debug_print(1, f"Loaded depth map (from imageio.v2) with: depth.shape=[{depth.shape}].")
            if depth is not None and depth.size > 0:
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                debug_print(1, f"Loaded depth map (from imageio.v2) with: depth.shape=[{depth.shape}].")
                return depth.astype(np.float32)
        except Exception:
            pass
        
        # imageio Fallback
        try:
            import imageio
            depth = imageio.imread(str(depth_path))
            debug_print(1, f"Loaded depth map (from imageio) with: depth.shape=[{depth.shape}].")
            if depth is not None and depth.size > 0:
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                debug_print(1, f"Loaded depth map (from imageio) with: depth.shape=[{depth.shape}].")
                return depth.astype(np.float32)
        except Exception:
            pass
        
        # exr package Fallback
        try:
            import exr
            depth = exr.read(str(depth_path))
            debug_print(1, f"Loaded depth map (from exr package) with: depth.shape=[{depth.shape}].")
            if depth is not None:
                if depth.ndim == 3:
                    depth = depth[:, :, 0]
                debug_print(1, f"Loaded depth map (from exr package) with: depth.shape=[{depth.shape}].")
                return depth.astype(np.float32)
        except ImportError:
            pass
        except Exception as e:
            debug_print(1, f"Exception: Failed to load depth map from exr package: [{e}].")
            pass
        
        debug_print(1, f"Warning: Could not load EXR file {depth_path.name}.")
        return None
        
    elif ext == '.png':
        depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        if depth is not None:
            debug_print(1, f"Loaded depth map (from PNG file) with: depth.shape=[{depth.shape}].")
            return depth.astype(np.float32)
    elif ext == '.npy':
        debug_print(1, f"Loaded depth map (from NumPy file) with: depth.shape=[{depth.shape}].")
        return np.load(str(depth_path))
    
    return None


def find_depth_file(dataset_root: Path, frame: Dict, depth_source: str = "mvsanywhere_depth") -> Optional[Path]:
    """
    Find corresponding depth file for a frame using metadata.
    
    Args:
        dataset_root: Root directory of the dataset
        frame: Frame dictionary from scene_meta.json
        depth_source: Key in frame dict for depth path (default: "mvsanywhere_depth")
                     Options: "mvsanywhere_depth", "moge2_depth", etc.
    
    Returns:
        Path to depth file if found, None otherwise
    """
    # Try to get depth path from frame metadata
    depth_path_rel = frame.get(depth_source)
    if depth_path_rel:
        depth_path = dataset_root / depth_path_rel
        if depth_path.exists():
            debug_print(1, f"Found depth file from metadata: [depth_path={depth_path}].")
            return depth_path
        else:
            debug_print(1, f"Depth path from metadata does not exist: [depth_path={depth_path}].")
    
    # Fallback: try alternative depth sources
    alternative_sources = ["moge2_depth", "mvsanywhere_depth"]
    for alt_source in alternative_sources:
        if alt_source == depth_source:
            continue
        depth_path_rel = frame.get(alt_source)
        if depth_path_rel:
            depth_path = dataset_root / depth_path_rel
            if depth_path.exists():
                debug_print(1, f"Found depth file from alternative source [{alt_source}]: [depth_path={depth_path}].")
                return depth_path
    
    # construct depth path from image path
    image_path_rel = frame.get('file_path') or frame.get('image')
    if image_path_rel:
        image_path = dataset_root / image_path_rel
        stem = image_path.stem
        
        depth_dir = dataset_root / "mvsanywhere" / "v0" / "depth"
        if depth_dir.exists():
            for ext in ['.exr', '.png', '.npy']:
                depth_path = depth_dir / f"{stem}{ext}"
                if depth_path.exists():
                    debug_print(1, f"Found depth file using legacy method: [depth_path={depth_path}].")
                    return depth_path
    
    debug_print(1, f"No depth file found for frame [frame_name={frame.get('frame_name', 'unknown')}].")
    return None


def load_scene_meta(dataset_root: Path) -> Dict:
    """Load scene metadata from scene_meta.json."""
    debug_print(1, f"Loading scene metadata from: [dataset_root={dataset_root / 'scene_meta.json'}]")
    meta_path = dataset_root / "scene_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"scene_meta.json not found at {meta_path}")
    
    with open(meta_path, 'r') as f:
        return json.load(f)


def parse_transform_matrix(transform: List) -> np.ndarray:
    """Parse 4x4 transformation matrix from flattened list."""
    debug_print(1, f"Parsing transformation matrix with: transform.length=[{len(transform)}].")
    return np.array(transform).reshape(4, 4)

def unproject_points(
    depth: np.ndarray,
    rgb: np.ndarray,
    intrinsics: Dict[str, float],
    transform_matrix: np.ndarray,
    depth_scale: float = 1.0,
    max_depth: float = 1000.0,
    downsample: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unproject depth map to 3D points using pinhole camera model.
    """
    debug_print(1, f"Unprojecting points with: \n\t-depth.shape=[{depth.shape}] \n\t-rgb.shape=[{rgb.shape}] \n\t-intrinsics=[{intrinsics}] \n\t-transform_matrix.shape=[{transform_matrix.shape}] \n\t-depth_scale=[{depth_scale}] \n\t-max_depth=[{max_depth}] \n\t-downsample=[{downsample}].")
    
    depth_h, depth_w = depth.shape
    rgb_h, rgb_w = rgb.shape[:2]
    
    # Resize depth if dimensions don't match
    if depth_h != rgb_h or depth_w != rgb_w:
        debug_print(2, f"Resizing depth to match RGB dimensions.")
        depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        debug_print(1, f"Resized depth to match RGB dimensions with: depth.shape=[{depth.shape}].")
    
    h, w = depth.shape
    if h != rgb_h or w != rgb_w:
        raise ValueError(f"Shape mismatch: depth ({h}, {w}) vs RGB ({rgb_h}, {rgb_w})")
    
    # Downsample
    if downsample > 1:
        debug_print(2, f"Downsampling depth and RGB.")
        depth = depth[::downsample, ::downsample]
        rgb = rgb[::downsample, ::downsample]
        h, w = depth.shape
        debug_print(1, f"Downsampled depth and RGB with: depth.shape=[{depth.shape}] rgb.shape=[{rgb.shape}].")
    
    # Check shape again
    rgb_h_ds, rgb_w_ds = rgb.shape[:2]
    if h != rgb_h_ds or w != rgb_w_ds:
        raise ValueError(f"Shape mismatch after downsampling: depth ({h}, {w}) vs RGB ({rgb_h_ds}, {rgb_w_ds})")
    
    # Get intrinsics
    fl_x = intrinsics.get('fl_x', intrinsics.get('fx', 0))
    fl_y = intrinsics.get('fl_y', intrinsics.get('fy', 0))
    cx = intrinsics.get('cx', w / 2.0)
    cy = intrinsics.get('cy', h / 2.0)
    
    # Scale intrinsics
    if downsample > 1:
        debug_print(1, f"Scaling intrinsics if downsampled. scale factor: [{downsample}].")
        debug_print(1, f"Scaling intrinsics if downsampled with: cx=[{cx}] cy=[{cy}] fl_x=[{fl_x}] fl_y=[{fl_y}].")
        cx /= downsample
        cy /= downsample
        fl_x /= downsample
        fl_y /= downsample
    
    # Coordinates
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Scale and filter depth
    depth_scaled = depth * depth_scale
    valid_mask = (depth_scaled > 0) & (depth_scaled < max_depth)
    
    # Unproject
    x_cam = (u - cx) * depth_scaled / fl_x
    y_cam = (v - cy) * depth_scaled / fl_y
    z_cam = depth_scaled
    
    points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)
    points_cam = points_cam[valid_mask]
    
    # Colors
    if rgb.ndim == 3:
        colors = rgb[valid_mask]
        debug_print(1, f"Indexed RGB with the valid mask with: \n\t-colors=[{colors.shape}].")
    elif rgb.ndim == 2:
        colors = np.stack([rgb[valid_mask], rgb[valid_mask], rgb[valid_mask]], axis=-1)
        debug_print(1, f"Indexed RGB with the valid mask with: \n\t-colors=[{colors.shape}].")
    else:
        debug_print(2, f"[Error] Unexpected RGB shape: {rgb.shape}")
        raise ValueError(f"Unexpected RGB shape: {rgb.shape}")
    
    # Transform to world
    points_cam_hom = np.concatenate([points_cam, np.ones((len(points_cam), 1))], axis=1)
    points_world_hom = (transform_matrix @ points_cam_hom.T).T
    points_world = points_world_hom[:, :3]
    
    debug_print(1, f"Transformed to world coordinates with: points_cam.shape=[{points_cam.shape}] points_world.shape=[{points_world.shape}] colors.shape=[{colors.shape}].")
    return points_world, colors


def calculate_quaternion_from_matrix(rotation_matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    Handles both scipy and manual fallback.
    """
    try:
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(rotation_matrix)
        wxyz = rotation.as_quat()  # Returns [x, y, z, w]
        wxyz = np.array([wxyz[3], wxyz[0], wxyz[1], wxyz[2]])  # Convert to [w, x, y, z]
        debug_print(1, f"Converted rotation matrix to quaternion: [{wxyz}].")
        return wxyz
    except ImportError:
        # manual quaternion conversion (simplified)
        debug_print(1, f"[Warning] Failed to convert rotation matrix to quaternion. Using manual conversion.")
        trace = np.trace(rotation_matrix)
        if trace > 0:
            s = np.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
            y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
            z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
        else:
            if rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                z = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
            elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
                s = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
                w = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / s
                x = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
            else:
                s = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
                w = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / s
                x = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / s
                y = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / s
                z = 0.25 * s
        return np.array([w, x, y, z])


def process_dataset_frames(dataset_root: Path, frames: List[Dict], frame_skip: int, global_intrinsics: Dict, depth_source: str = "mvsanywhere_depth") -> List[Dict]:
    """
    Iterate through frames and load images, depth maps, and metadata.
    
    Args:
        dataset_root: Root directory of the dataset
        frames: List of frame dictionaries from scene_meta.json
        frame_skip: Process every Nth frame
        global_intrinsics: Global camera intrinsics from scene metadata
        depth_source: Key in frame dict for depth path (default: "mvsanywhere_depth")
    """
    processed_frames = []
    
    # Process frames with skip
    target_frames = frames[::frame_skip]
    debug_print(1, f"Found [{len(frames)}] frames. Processing every [{frame_skip}] frame(s) -> {len(target_frames)} frames.")

    for i, frame in enumerate(target_frames):
        debug_print(3, f"Processing frame [{i+1}/{len(target_frames)}].")
        
        # Get image path - try both 'file_path' and 'image' fields
        image_path_rel = frame.get('file_path') or frame.get('image')
        if not image_path_rel:
            debug_print(3, f"[Warning] No file_path or image in frame [{i}], skipping.")
            continue
        
        image_path = dataset_root / image_path_rel
        if not image_path.exists():
            debug_print(1, f"[Warning] Image not found: [{image_path}], skipping.")
            continue
        
        # Load RGB image
        rgb = cv2.imread(str(image_path))
        if rgb is None:
            debug_print(1, f"[Warning] Failed to load image: [{image_path}], skipping.")
            continue
        
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        
        # Find and load depth map using frame metadata
        depth_path = find_depth_file(dataset_root, frame, depth_source=depth_source)
        if depth_path is None:
            debug_print(1, f"[Warning] No depth map found for frame [{frame.get('frame_name', i)}], skipping.")
            continue
        
        depth = load_depth_map(depth_path)
        if depth is None:
            debug_print(1, f"[Warning] Failed to load depth map: [{depth_path}], skipping.")
            continue
        
        # get intrinsics
        intrinsics = {
            'fl_x': frame.get('fl_x', global_intrinsics.get('fl_x', 0)),
            'fl_y': frame.get('fl_y', global_intrinsics.get('fl_y', 0)),
            'cx': frame.get('cx', global_intrinsics.get('cx', 0)),
            'cy': frame.get('cy', global_intrinsics.get('cy', 0)),
        }
        
        # Get transformation matrix
        transform = frame.get('transform_matrix', None)
        if transform is None:
            debug_print(1, f"[Warning] No transform_matrix in frame [{i}], skipping.")
            continue
        
        transform_matrix = parse_transform_matrix(transform)
        
        # Store frame data
        processed_frames.append({
            'rgb': rgb,
            'depth': depth,
            'intrinsics': intrinsics,
            'transform_matrix': transform_matrix,
            'image_path': image_path,
            'frame_name': frame.get('frame_name', f'frame_{i}'),
        })
    
    debug_print(1, f"Successfully processed [{len(processed_frames)}] frames.")
    return processed_frames


def render_point_clouds(
    server: viser.ViserServer, 
    point_cloud_handles: List, 
    processed_frames: List[Dict], 
    depth_scale: float, 
    max_depth: float, 
    point_size: float, 
    downsample_factor: int
):
    """Generates and adds point clouds to the Viser scene."""
    # Clear existing point clouds
    for handle in point_cloud_handles:
        try:
            server.scene.remove(handle)
            debug_print(1, f"Removed point cloud handle: [{handle}].")
        except (AttributeError, TypeError):
            debug_print(1, f"[Error] Failed to remove point cloud handle: [{handle}].")
            try:
                server.scene.delete(handle)
                debug_print(1, f"Deleted point cloud handle: [{handle}].")
            except (AttributeError, TypeError):
                debug_print(1, f"[Error] Failed to delete point cloud handle: [{handle}].")
                pass
    point_cloud_handles.clear()
    
    debug_print(1, f"Updating point clouds with: \n\t-depth_scale=[{depth_scale:.3f}] \n\t-max_depth=[{max_depth:.1f}] \n\t-point_size=[{point_size:.3f}].")
    
    all_points = []
    all_colors = []
    
    for frame_data in processed_frames:
        points, colors = unproject_points(
            frame_data['depth'],
            frame_data['rgb'],
            frame_data['intrinsics'],
            frame_data['transform_matrix'],
            depth_scale=depth_scale,
            max_depth=max_depth,
            downsample=downsample_factor,
        )
        
        if len(points) > 0:
            all_points.append(points)
            all_colors.append(colors)
    
    if len(all_points) > 0:
        # concatenate all points
        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        
        # normalize colors to [0, 1]
        all_colors = all_colors.astype(np.float32) / 255.0
        
        # add point cloud to scene with fallbacks
        try:
            handle = server.scene.add_point_cloud(
                name=f"point_cloud",
                points=all_points,
                colors=all_colors,
                point_size=point_size,
            )
        except (TypeError, AttributeError):
            try:
                debug_print(1, f"[ALTERNATIVE] Adding point cloud with: \n\t-points=[{all_points.shape}] \n\t-colors=[{all_colors.shape}] \n\t-point_size=[{point_size:.3f}].")
                handle = server.scene.add_point_cloud(
                    points=all_points,
                    colors=all_colors,
                    point_size=point_size,
                )
            except (TypeError, AttributeError):
                debug_print(1, f"[MINIMAL] Adding point cloud with: \n\t-points=[{all_points.shape}] \n\t-colors=[{all_colors.shape}] \n\t-point_size=[{point_size:.3f}].")
                handle = server.scene.add_point_cloud(
                    all_points,
                    colors=all_colors,
                )
        
        point_cloud_handles.append(handle)
        debug_print(1, f"Added point cloud with [{len(all_points)}] points.")


def render_cameras(server: viser.ViserServer, camera_handles: List, processed_frames: List[Dict]):
    """Generates and adds camera frustums to the Viser scene."""
    # Clear existing cameras
    for handle in camera_handles:
        try:
            server.scene.remove(handle)
            debug_print(1, f"Removed camera handle: [{handle}].")
        except (AttributeError, TypeError):
            try:
                server.scene.delete(handle)
                debug_print(1, f"Deleted camera handle: [{handle}].")
            except (AttributeError, TypeError):
                debug_print(1, f"[Error] Failed to delete camera handle: [{handle}].")
                pass
    camera_handles.clear()
    debug_print(1, f"Adding camera frustums.")
    
    for i, frame_data in enumerate(processed_frames):
        debug_print(3, f"Adding camera frustum [{i+1}/{len(processed_frames)}].")
        rgb = frame_data['rgb']
        intrinsics = frame_data['intrinsics']
        transform_matrix = frame_data['transform_matrix']
        
        h, w = rgb.shape[:2]
        
        # get intrinsics
        fl_x = intrinsics.get('fl_x', intrinsics.get('fx', 0))
        cx = intrinsics.get('cx', w / 2.0)
        
        # Viser expects camera-to-world transform
        position = transform_matrix[:3, 3]
        rotation_matrix = transform_matrix[:3, :3]
        
        # Get quaternion
        wxyz = calculate_quaternion_from_matrix(rotation_matrix)
        
        # calculate FOV
        fov = 2 * np.arctan(w / (2 * fl_x)) if fl_x > 0 else np.pi / 3
        aspect = w / h if h > 0 else 1.0
        
        # add camera frustum with different API attempts
        handle = None
        try:
            debug_print(1, f"[ALTERNATIVE] Adding camera frustum.")
            debug_print(3, f"[ALTERNATIVE] Adding camera frustum with: \n\t-name=[camera_{i}] \n\t-fov=[{fov}] \n\t-aspect=[{aspect}] \n\t-scale=[0.1] \n\t-wxyz=[{wxyz}] \n\t-position=[{position}].")
            handle = server.scene.add_camera_frustum(
                name=f"camera_{i}",
                fov=fov,
                aspect=aspect,
                scale=0.1,
                wxyz=wxyz,
                position=position,
                image=rgb,
            )
        except (TypeError, AttributeError):
            try:
                debug_print(1, f"[MINIMAL] Adding camera frustum.")
                debug_print(3, f"[MINIMAL] Adding camera frustum with: \n\t-fov=[{fov}] \n\t-aspect=[{aspect}] \n\t-scale=[0.1] \n\t-wxyz=[{wxyz}] \n\t-position=[{position}].")
                handle = server.scene.add_camera_frustum(
                    fov=fov,
                    aspect=aspect,
                    scale=0.1,
                    wxyz=wxyz,
                    position=position,
                    image=rgb,
                )
            except (TypeError, AttributeError):
                try:
                    debug_print(1, f"[MINIMAL] Adding camera frustum.")
                    debug_print(3, f"[MINIMAL] Adding camera frustum with: \n\t-transform=[{transform_matrix}].")
                    handle = server.scene.add_camera_frustum(
                        transform=transform_matrix,
                        image=rgb,
                    )
                except (TypeError, AttributeError) as e:
                    debug_print(1, f"[Error] Failed to add camera frustum for frame [{i}]: [{e}].")
                    handle = None
        
        if handle is not None:
            camera_handles.append(handle)
    
    debug_print(1, f"Added [{len(camera_handles)}] camera frustums.")


def get_parser():
    """get parser for command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                    help="Path to dataset root directory")
    parser.add_argument("--port", type=int, default=8080,
                    help="Port for Viser server (default: 8080)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                    help="Host for Viser server (default: 0.0.0.0)")
    parser.add_argument("--frame_skip", type=int, default=1,
                    help="Process every Nth frame (default: 1, process all)")
    parser.add_argument("--downsample", type=int, default=4,
                    help="Downsample factor for point cloud (default: 4)")
    parser.add_argument("--default_depth_scale", type=float, default=1.0,
                    help="Default depth scale (default: 1.0)")
    parser.add_argument("--default_max_depth", type=float, default=100.0,
                    help="Default max depth cutoff (default: 100.0)")
    parser.add_argument("--default_point_size", type=float, default=0.01,
                        help="Default point size (default: 0.01)")
    parser.add_argument("--dataset", type=str, default="mvsanywhere",
                        choices=["mvsanywhere", "moge"],
                        help="Dataset type to use: 'mvsanywhere' or 'moge' (default: mvsanywhere)")
    parser.add_argument("--depth_source", type=str, default=None,
                        choices=["mvsanywhere_depth", "moge2_depth"],
                        help="Depth source key to use from scene_meta.json. If not provided, will be inferred from --dataset")
    parser.add_argument("--debug", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Debug level (default: 0)")
    parser.add_argument("--log_filepath", type=str, default=None,
                        help="Path to log file (default: None)")
    return parser


def main():
    parser = get_parser()    
    args = parser.parse_args()
    
    setup_logging(args.log_filepath, args)

    global DEBUG_LEVEL
    DEBUG_LEVEL = args.debug 
    debug_print(1, f"DEBUG_LEVEL: [{DEBUG_LEVEL}].")

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    
    # 1. Load scene metadata
    scene_meta = load_scene_meta(dataset_root)
    
    # Extract global intrinsics (used when shared_intrinsics is true)
    global_intrinsics = {
        'fl_x': scene_meta.get('fl_x', 0),
        'fl_y': scene_meta.get('fl_y', 0),
        'cx': scene_meta.get('cx', 0),
        'cy': scene_meta.get('cy', 0),
    }
    
    # Log scene metadata info
    debug_print(1, f"Scene metadata loaded: scene_name=[{scene_meta.get('scene_name')}], "
                   f"shared_intrinsics=[{scene_meta.get('shared_intrinsics', False)}], "
                   f"camera_model=[{scene_meta.get('camera_model')}]")
    
    frames = scene_meta.get('frames', [])
    if not frames:
        raise ValueError("No frames found in scene_meta.json")
    
    # Determine depth source from dataset argument or explicit depth_source
    if args.depth_source:
        depth_source = args.depth_source
        debug_print(1, f"Using explicit depth_source: [{depth_source}]")
    else:
        # Map dataset to depth_source key
        dataset_to_depth_source = {
            "mvsanywhere": "mvsanywhere_depth",
            "moge": "moge2_depth"
        }
        depth_source = dataset_to_depth_source.get(args.dataset, "mvsanywhere_depth")
        debug_print(1, f"Using dataset [{args.dataset}] -> depth_source: [{depth_source}]")
    
    # 2. Process all frames
    processed_frames = process_dataset_frames(
        dataset_root, 
        frames, 
        args.frame_skip, 
        global_intrinsics,
        depth_source=depth_source
    )
    
    # 3. Initialize Viser server
    try:
        server = viser.ViserServer(host=args.host, port=args.port)
    except TypeError:
        server = viser.ViserServer()
        debug_print(1, f"Failed to initialize Viser server with: \n\t-host: [{args.host}] \n\tport: [{args.port}]. \n\tUsing default host: [0.0.0.0] and port: [8080].")
    
    # 4. Setup GUI controls
    depth_scale_slider = server.gui.add_slider("Depth Scale", 0.001, 1000.0, 0.001, args.default_depth_scale)
    point_size_slider = server.gui.add_slider("Point Size", 0.001, 0.1, 0.001, args.default_point_size)
    max_depth_slider = server.gui.add_slider("Max Depth", 0.1, 1000.0, 0.1, args.default_max_depth)
    
    # Handle lists
    point_cloud_handles = []
    camera_handles = []
    
    # 5. Define Wrapper for updates (links GUI state to Logic functions)
    def update_visualization():
        # Read current slider values
        try:
            depth_scale = depth_scale_slider.value
            max_depth = max_depth_slider.value
            point_size = point_size_slider.value
            debug_print(1, f"Using values from sliders: depth_scale=[{depth_scale:.3f}], max_depth=[{max_depth:.1f}], point_size=[{point_size:.3f}].")
        except AttributeError:
            debug_print(1, f"[Warning] Failed to get value from sliders. Using default values.")
            depth_scale = args.default_depth_scale
            max_depth = args.default_max_depth
            point_size = args.default_point_size
            
        render_point_clouds(
            server=server,
            point_cloud_handles=point_cloud_handles,
            processed_frames=processed_frames,
            depth_scale=depth_scale,
            max_depth=max_depth,
            point_size=point_size,
            downsample_factor=args.downsample
        )

    # 6. Initial Render
    update_visualization()
    render_cameras(server, camera_handles, processed_frames)
    
    # 7. Setup Callbacks
    try:
        @depth_scale_slider.on_update
        def _(_): update_visualization()
        
        @max_depth_slider.on_update
        def _(_): update_visualization()
        
        @point_size_slider.on_update
        def _(_): update_visualization()
    except AttributeError:
        try:
            depth_scale_slider.on_change(lambda _: update_visualization())
            max_depth_slider.on_change(lambda _: update_visualization())
            point_size_slider.on_change(lambda _: update_visualization())
        except AttributeError:
            print("\n[Warning] Could not set up slider callbacks. Sliders may not update visualization.")
    
    print("\n" + "="*50)
    print("-- Viser server is running --")
    print(f"Local URL: http://localhost:{args.port}")
    print(f"Remote URL: http://{args.host}:{args.port}")
    print("="*50 + "\n")
    
    # Keep server running
    try:
        try:
            server.start()
        except (AttributeError, TypeError):
            pass
        input("Press Enter to stop the server...\n")
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        close_logging()
        try:
            server.close()
        except (AttributeError, TypeError):
            pass


if __name__ == "__main__":
    main()