import torch
import logging

from .coordinates import convert_points_to_homogeneous


logger = logging.getLogger(__name__)


def unproject_depth_map_to_world_points(
    depth, intrinsics, c2w=None, ixt_normalized=False, num_patches_x=None, num_patches_y=None
):
    logger.debug("depth", extra={"depth": depth.shape})
    logger.debug("intrinsics", extra={"intrinsics": intrinsics.shape})
    logger.debug("c2w", extra={"c2w": c2w.shape})
    logger.debug("ixt_normalized", extra={"ixt_normalized": ixt_normalized})
    logger.debug("num_patches_x", extra={"num_patches_x": num_patches_x})
    logger.debug("num_patches_y", extra={"num_patches_y": num_patches_y})
    if c2w is None:
        logger.debug("c2w is None", extra={"c2w": c2w})
        c2w = torch.eye(4, device=depth.device, dtype=depth.dtype)
        if depth.ndim > 2:
            batch_dims = depth.shape[:-2]
            logger.debug("batch_dims", extra={"batch_dims": batch_dims})
            c2w = c2w.unsqueeze(0).expand(*batch_dims, 4, 4)
            logger.debug("c2w", extra={"c2w": c2w.shape})
    if not ixt_normalized:
        h, w = depth.shape[-2], depth.shape[-1]
        logger.debug("h", extra={"h": h})
        logger.debug("w", extra={"w": w})
        x_grid, y_grid = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )
        logger.debug("x_grid", extra={"x_grid": x_grid.shape})
        logger.debug("y_grid", extra={"y_grid": y_grid.shape})
    else:
        if num_patches_x is None or num_patches_y is None:
            logger.error(
                "num_patches_x and num_patches_y must be provided when ixt_normalized=True",
                extra={"num_patches_x": num_patches_x, "num_patches_y": num_patches_y},
            )
            raise ValueError("num_patches_x and num_patches_y must be provided when ixt_normalized=True")
        logger.debug("dx", extra={"dx": dx})
        logger.debug("dy", extra={"dy": dy})
        logger.debug("max_y", extra={"max_y": max_y})
        logger.debug("min_y", extra={"min_y": min_y})
        logger.debug("max_x", extra={"max_x": max_x})
        logger.debug("min_x", extra={"min_x": min_x})
        logger.debug("grid_shift", extra={"grid_shift": grid_shift})
        dx = 1 / num_patches_x
        dy = 1 / num_patches_y
        max_y = 1 - dy
        min_y = -max_y
        max_x = 1 - dx
        min_x = -max_x
        grid_shift = 1.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(
                min_y + grid_shift,
                max_y + grid_shift,
                num_patches_y,
                dtype=torch.float32,
                device=depth.device,
            ),
            torch.linspace(
                min_x + grid_shift,
                max_x + grid_shift,
                num_patches_x,
                dtype=torch.float32,
                device=depth.device,
            ),
            indexing="ij",
        )
        logger.debug("y_grid", extra={"y_grid": y_grid.shape})
        logger.debug("x_grid", extra={"x_grid": x_grid.shape})
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (H, W, 2)
    logger.debug("pixel_space_points", extra={"pixel_space_points": pixel_space_points.shape})
    if depth.ndim > 2:
        batch_dims = depth.shape[:-2]  # (...,)
        for _ in range(len(batch_dims)):
            pixel_space_points = pixel_space_points.unsqueeze(0)
        pixel_space_points = pixel_space_points.expand(*batch_dims, *pixel_space_points.shape[-2:])
        logger.debug("pixel_space_points", extra={"pixel_space_points": pixel_space_points.shape})
    camera_points = convert_pixel_coordinates_to_camera_space(pixel_space_points, depth, intrinsics)
    world_points = transform_camera_points_to_world_space(camera_points, c2w)
    return world_points


def convert_pixel_coordinates_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space 3D points.

    Args:
        pixel_space_points: Pixel coordinates of shape (..., H, W, 2)
        depth: Depth values of shape (..., H, W)
        intrinsics: Camera intrinsics matrix of shape (..., 3, 3)

    Returns:
        Camera space 3D points of shape (..., H, W, 3)
    """
    logger.debug("pixel_space_points", extra={"pixel_space_points": pixel_space_points.shape})
    logger.debug("depth", extra={"depth": depth.shape})
    logger.debug("intrinsics", extra={"intrinsics": intrinsics.shape})
    pixel_homogeneous = convert_points_to_homogeneous(pixel_space_points)

    batch_shape = pixel_homogeneous.shape[:-2]
    spatial_shape = pixel_homogeneous.shape[-2:]

    pixel_flat = pixel_homogeneous.reshape(*batch_shape, -1, 3)

    intrinsics_inv = torch.linalg.inv(intrinsics)
    normalized_camera_coords = (intrinsics_inv @ pixel_flat.transpose(-2, -1)).transpose(-2, -1)

    normalized_camera_coords = normalized_camera_coords.reshape(*batch_shape, *spatial_shape, 3)

    depth_expanded = depth.unsqueeze(-1)
    camera_points = normalized_camera_coords * depth_expanded
    logger.debug("Camera points", extra={"camera_points": camera_points.shape})
    return camera_points


def transform_camera_points_to_world_space(camera_points, c2w):
    logger.debug("camera_points", extra={"camera_points": camera_points.shape})
    logger.debug("c2w", extra={"c2w": c2w.shape})
    logger.debug("camera_points", extra={"camera_points": camera_points.shape})
    logger.debug("c2w", extra={"c2w": c2w.shape})
    camera_points_homogeneous = convert_points_to_homogeneous(camera_points)

    batch_shape = camera_points_homogeneous.shape[:-2]
    spatial_shape = camera_points_homogeneous.shape[-2:]

    camera_flat = camera_points_homogeneous.reshape(*batch_shape, -1, 4)
    logger.debug("camera_flat", extra={"camera_flat": camera_flat.shape})

    world_flat = (c2w @ camera_flat.transpose(-2, -1)).transpose(-2, -1)
    logger.debug("world_flat", extra={"world_flat": world_flat.shape})

    world_homogeneous = world_flat.reshape(*batch_shape, *spatial_shape, 4)

    world_points = world_homogeneous[..., :3]
    logger.debug("World points", extra={"world_points": world_points.shape})

    return world_points
