from abc import ABC, abstractmethod
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any

from .util import get_da3_metric_depth
from .logger import debug_print
from .geometry import parse_transform_matrix

class BaseDataset(ABC):
    """Abstract Base Class for dataset loaders."""
    
    def __init__(self, root: Path, frame_skip: int = 1):
        self.root = root
        self.frame_skip = frame_skip
        self.frames = []
        
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
            
        self._load_metadata()
        self._process_frames()

    @abstractmethod
    def _load_metadata(self):
        """Load scene metadata (intrinsics, frames list)."""
        pass

    @abstractmethod
    def _process_frames(self):
        """Process and store frame data (paths, transforms)."""
        pass

    def get_frames(self) -> List[Dict]:
        return self.frames

    def _load_depth_file(self, depth_path: Path) -> Optional[np.ndarray]:
        """Helper to load depth from .exr, .png, .npy"""
        if not depth_path.exists():
            return None
        
        ext = depth_path.suffix.lower()
        try:
            if ext == '.exr':
                import OpenEXR, Imath
                exr_file = OpenEXR.InputFile(str(depth_path))
                dw = exr_file.header()['dataWindow']
                h, w = dw.max.y - dw.min.y + 1, dw.max.x - dw.min.x + 1
                for c in ['R', 'Y', 'Z', 'G']:
                    try:
                        d = np.frombuffer(exr_file.channel(c, Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
                        debug_print(3, f"Loaded depth from {depth_path} with shape {d.shape}")
                        return d.reshape((h, w))
                    except: continue
            elif ext == '.png':
                d = cv2.imread(str(depth_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
                debug_print(3, f"Loaded depth from {depth_path} with shape {d.shape}")
                return d.astype(np.float32) if d is not None else None
            elif ext == '.npy':
                d = np.load(str(depth_path))
                debug_print(3, f"Loaded depth from {depth_path} with shape {d.shape}")
                return d
        except Exception as e:
            debug_print(1, f"Failed to load depth {depth_path}: {e}")
        return None


# class NerfStudioDataset(BaseDataset):
class WAI_Dataset(BaseDataset):

    def __init__(self, root: Path, frame_skip: int, depth_source_key: str = "mvsanywhere_depth"):
        self.depth_source_key = depth_source_key
        self.global_intrinsics = {}
        super().__init__(root, frame_skip)

    def _load_metadata(self):
        meta_path = self.root / "scene_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"scene_meta.json not found at {meta_path}")
        
        # self.meta contains the json data as a dictionary
        with open(meta_path, 'r') as f:
            self.meta = json.load(f)
            
        self.global_intrinsics = {
            'fl_x': self.meta.get('fl_x', 0),
            'fl_y': self.meta.get('fl_y', 0),
            'cx': self.meta.get('cx', 0),
            'cy': self.meta.get('cy', 0),
        }
        debug_print(1, f"Global intrinsics: {self.global_intrinsics}")
        debug_print(3, f"Metadata loaded. Shared intrinsics: {self.meta.get('shared_intrinsics', False)}")

    def _find_depth_path(self, frame: Dict) -> Optional[Path]:
        # Priority 1: Requested Source - Most likely to run this
        if frame.get(self.depth_source_key):
            p = self.root / frame.get(self.depth_source_key)
            if p.exists(): 
                debug_print(2, f"[DEPTH:PRIORITY 1] Found depth path: {p}")
                return p
        
        # Priority 2: Alternatives
        for alt in ["moge2_depth", "mvsanywhere_depth"]:
            if alt != self.depth_source_key and frame.get(alt):
                p = self.root / frame.get(alt)
                if p.exists(): 
                    debug_print(2, f"[DEPTH:PRIORITY 2] Found depth path: {p}")
                    return p
        
        # Priority 3: Legacy filename matching
        img_rel = frame.get('file_path') or frame.get('image')
        if img_rel:
            stem = Path(img_rel).stem
            for ext in ['.exr', '.png', '.npy']:
                p = self.root / "mvsanywhere" / "v0" / "depth" / f"{stem}{ext}"
                if p.exists(): 
                    debug_print(2, f"[DEPTH:PRIORITY 3] Found depth path: {p}")
                    return p
        debug_print(1, f"[DEPTH:None] No depth path found")
        return None

    def _process_frames(self):
        raw_frames = self.meta.get('frames', [])
        target_frames = raw_frames[::self.frame_skip]
        debug_print(1, f"Processing {len(target_frames)} frames (Skip: {self.frame_skip}).")

        for i, frame in enumerate(target_frames):
            # get image path
            img_rel = frame.get('file_path') or frame.get('image')
            if not img_rel: continue
            
            img_path = self.root / img_rel
            if not img_path.exists(): continue
            
            rgb = cv2.imread(str(img_path))
            if rgb is None: continue
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            # get depth path
            depth_path = self._find_depth_path(frame)
            if not depth_path: continue
            
            depth = self._load_depth_file(depth_path)
            if depth is None: continue

            # get intrinsics
            intrinsics = {
                'fl_x': frame.get('fl_x', self.global_intrinsics['fl_x']),
                'fl_y': frame.get('fl_y', self.global_intrinsics['fl_y']),
                'cx': frame.get('cx', self.global_intrinsics['cx']),
                'cy': frame.get('cy', self.global_intrinsics['cy']),
            }
            
            transform = frame.get('transform_matrix')
            if not transform: continue

            self.frames.append({
                'rgb': rgb,
                'depth': depth,
                'intrinsics': intrinsics,
                'transform_matrix': parse_transform_matrix(transform),
                'frame_name': frame.get('frame_name', f'frame_{i}'),
                'id': i
            })
            debug_print(3, f"Processed frame {i}")


class DenseDataset(BaseDataset):
    def __init__(self, root: Path, frame_skip: int, da3_scaling: bool = False):
        self.da3_scaling = da3_scaling
        super().__init__(root, frame_skip)

    def _load_metadata(self):
        # no JSON; structure validated during _process_frames
        pass

    def _process_frames(self):
        # Support both direct structure and 'dense' subdirectory
        if (self.root / "dense" / "rgb").exists():
            base_dir = self.root / "dense"
            debug_print(1, "Found dense subdirectory structure")
        else:
            base_dir = self.root
            debug_print(1, "Using direct structure (no dense subdirectory)")
        
        rgb_dir = base_dir / "rgb"
        depth_dir = base_dir / "depth"
        cam_dir = base_dir / "cam"

        if not rgb_dir.exists():
            debug_print(0, f"RGB directory not found: {rgb_dir}")
            return

        rgb_files = sorted(list(rgb_dir.glob("*.png")))[::self.frame_skip]
        debug_print(1, f"Processing {len(rgb_files)} frames from dense structure.")

        for i, rgb_path in enumerate(rgb_files):
            stem = rgb_path.stem
            depth_path = depth_dir / f"{stem}.npy"
            cam_path = cam_dir / f"{stem}.npz"

            if not depth_path.exists():
                debug_print(2, f"Skipping {stem}: depth file not found")
                continue
            if not cam_path.exists():
                debug_print(2, f"Skipping {stem}: cam file not found")
                continue

            try:
                # RGB & Depth
                rgb = cv2.imread(str(rgb_path))
                if rgb is None:
                    debug_print(1, f"Failed to load RGB: {rgb_path}")
                    continue
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                
                depth = np.load(str(depth_path))
                if depth is None:
                    debug_print(1, f"Failed to load depth: {depth_path}")
                    continue
                
                # Validate depth shape matches RGB
                if depth.shape[:2] != rgb.shape[:2]:
                    debug_print(1, f"Shape mismatch: RGB {rgb.shape[:2]} vs Depth {depth.shape[:2]} for {stem}")
                    # Will be resized in unproject_points, but log the issue

                # Camera Data
                cam_data = np.load(str(cam_path))
                
                if 'intrinsic' not in cam_data:
                    debug_print(1, f"Missing 'intrinsic' key in {cam_path}")
                    continue
                if 'extrinsic' not in cam_data:
                    debug_print(1, f"Missing 'extrinsic' key in {cam_path}")
                    continue
                
                # Intrinsic (3, 3)
                k = cam_data['intrinsic']
                if k.shape != (3, 3):
                    debug_print(1, f"Unexpected intrinsic shape: {k.shape} for {stem}")
                    continue
                    
                intrinsics = {
                    'fl_x': float(k[0, 0]),
                    'fl_y': float(k[1, 1]),
                    'cx': float(k[0, 2]),
                    'cy': float(k[1, 2]),
                }

                # Apply DA3METRIC-LARGE conversion
                if self.da3_scaling:
                    depth = get_da3_metric_depth(depth, intrinsics)

                # Extrinsic handling: support both (3, 4) and (4, 4)
                ext = cam_data['extrinsic']
                if ext.shape == (4, 4):
                    transform = ext.astype(np.float32)
                    debug_print(3, f"Using 4x4 extrinsic matrix for {stem}")
                elif ext.shape == (3, 4):
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, :4] = ext.astype(np.float32)
                    debug_print(3, f"Using 3x4 extrinsic matrix for {stem}")
                else:
                    debug_print(1, f"Unexpected extrinsic shape: {ext.shape} for {stem}")
                    continue
                
                # invert to camera-to-world
                transform = np.linalg.inv(transform)
                
                debug_print(2, f"Processed frame {i} ({stem}): RGB={rgb.shape}, Depth={depth.shape}, Intrinsics={intrinsics}")

                self.frames.append({
                    'rgb': rgb,
                    'depth': depth,
                    'intrinsics': intrinsics,
                    'transform_matrix': transform,
                    'frame_name': stem,
                    'id': i
                })
            except Exception as e:
                debug_print(1, f"Error processing frame {stem}: {e}")
                import traceback
                debug_print(2, traceback.format_exc())
                continue

