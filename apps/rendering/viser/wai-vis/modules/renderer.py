import viser
import numpy as np
from typing import List, Dict
from .logger import debug_print
from .geometry import unproject_points, calculate_quaternion_from_matrix

class ViserApp:
    def __init__(self, dataset, host: str, port: int, downsample: int, defaults: Dict, label: str = "main_point_cloud"):
        """
        Args:
            dataset: Instance of BaseDataset
            host, port: Server config
            downsample: Point cloud downsample factor
            defaults: Dict with 'depth_scale', 'point_size', 'max_depth'
        """
        self.dataset = dataset
        self.downsample = downsample
        
        # Initialize Server
        try:
            self.server = viser.ViserServer(host=host, port=port, label=label)
        except Exception:
            debug_print(1, f"Server init failed, trying default {defaults['host']}:{defaults['port']}") # 0.0.0.0:8080
            self.server = viser.ViserServer()

        self.server.gui.configure_theme(dark_mode=True)

        self.progress_bar_handle_1 = self.server.gui.add_progress_bar(value=0.0)
        self.progress_bar_handle_2 = self.server.gui.add_progress_bar(value=0.0)

        # GUI Controls
        self.slider_depth_scale = self.server.gui.add_slider(
            label="Depth Scale", min=0.001, max=1000.0, step=0.001, initial_value=defaults['depth_scale']
        )
        self.slider_point_size = self.server.gui.add_slider(
            label="Point Size", min=0.001, max=0.1, step=0.001, initial_value=defaults['point_size']
        )
        self.slider_max_depth = self.server.gui.add_slider(
            label="Max Depth", min=0.1, max=1000.0, step=0.1, initial_value=defaults['max_depth']
        )
        
        # State handles
        self.pc_handles = []
        self.cam_handles = []
        self.tc_handles = []
        
        # Bind Callbacks
        self.slider_depth_scale.on_update(self.update_visualization)
        self.slider_point_size.on_update(self.update_visualization)
        self.slider_max_depth.on_update(self.update_visualization)


    def render_cameras(self):
        """One-time render of camera frustums."""
        debug_print(1, "Rendering camera frustums...")
        
        # Clean existing
        for h in self.cam_handles:
            self.server.scene.remove_by_name(h.name)
        self.cam_handles.clear()

        total_frames = len(self.dataset.get_frames())

        for frame in self.dataset.get_frames():
            rgb = frame['rgb']
            h, w = rgb.shape[:2]
            intrinsics = frame['intrinsics']
            fl_x = intrinsics.get('fl_x', 0)
            
            # Geometry
            transform = frame['transform_matrix']
            pos = transform[:3, 3]
            quat = calculate_quaternion_from_matrix(transform[:3, :3])
            
            fov = 2 * np.arctan(w / (2 * fl_x)) if fl_x > 0 else np.pi / 3
            aspect = w / h if h > 0 else 1.0

            try:
                handle = self.server.scene.add_camera_frustum(
                    name=f"camera_{frame['id']}",
                    fov=fov,
                    aspect=aspect,
                    scale=0.1,
                    wxyz=quat,
                    position=pos,
                    image=rgb
                )
                self.cam_handles.append(handle)
                debug_print(1, f"Added camera {frame['id']}: RGB-SHAPE: {rgb.shape}")
                debug_print(3, f"Camera {frame['id']} position: {pos}, quaternion: {quat}")
            except Exception as e:
                debug_print(1, f"Failed to add camera {frame['id']}: {e}")
        
            # experiment:
            # try: 
            #     tc_handle = self.server.scene.add_transform_controls(
            #         name=f"camera_{frame['id']}",
            #         scale=0.1,
            #         line_width=0.01,
            #         fixed=False,
            #         active_axes=(True, True, True),
            #         disable_axes=False,
            #         disable_sliders=False,
            #         disable_rotations=False,
            #         translation_limits=((-1000, 1000), (-1000, 1000), (-1000, 1000)),
            #         rotation_limits=((-1000, 1000), (-1000, 1000), (-1000, 1000)),
            #         depth_test=True,
            #         opacity=1.0,
            #         wxyz=quat,
            #         position=pos,
            #         visible=True
            #     )
            #     self.tc_handles.append(tc_handle)
            #     debug_print(2, f"Added transform controls for camera {frame['id']}")
            # except Exception as e:
            #     debug_print(1, f"Failed to add transform controls for camera {frame['id']}: {e}")
            
            self.progress_bar_handle_1.value = ((frame['id'] + 1) / total_frames) * 100.0


    def update_visualization(self, _=None):
        """Callback to regenerate point clouds based on slider values."""
        # Read sliders
        d_scale = self.slider_depth_scale.value
        pt_size = self.slider_point_size.value
        max_d = self.slider_max_depth.value
        
        debug_print(1, f"Update: scale={d_scale}, size={pt_size}, max_d={max_d}")

        # Clean old clouds
        for h in self.pc_handles:
            self.server.scene.remove_by_name(h.name)
        self.pc_handles.clear()

        all_points = []
        all_colors = []
        total_frames = len(self.dataset.get_frames())
        # Generate new clouds
        for frame in self.dataset.get_frames():
            pts, cols = unproject_points(
                frame['depth'],
                frame['rgb'],
                frame['intrinsics'],
                frame['transform_matrix'],
                depth_scale=d_scale,
                max_depth=max_d,
                downsample=self.downsample
            )
            if len(pts) > 0:
                all_points.append(pts)
                all_colors.append(cols)
                debug_print(1, f"Added point cloud {frame['id']}: RGB-SHAPE: {frame['rgb'].shape}, DEPTH-SHAPE: {frame['depth'].shape}")
            
            self.progress_bar_handle_2.value = ((frame['id'] + 1) / total_frames) * 100.0

        # Merge and Add to Scene
        if all_points:
            points = np.concatenate(all_points, axis=0)
            colors = np.concatenate(all_colors, axis=0).astype(np.float32) / 255.0
            
            try:
                handle = self.server.scene.add_point_cloud(
                    name="main_point_cloud",
                    points=points,
                    colors=colors,
                    point_size=pt_size,
                )
                self.pc_handles.append(handle)
                debug_print(1, f"Rendered cloud with {len(points)} points.")
            except Exception as e:
                debug_print(1, f"Error adding point cloud: {e}")

    def run(self):
        print(f"\nViser running at http://{self.server.get_host()}:{self.server.get_port()}\n")
        
        self.update_visualization()
        self.render_cameras()
        
        try:
            input("Press Enter to stop the server...\n")
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.server.stop()
            except AttributeError:
                pass

