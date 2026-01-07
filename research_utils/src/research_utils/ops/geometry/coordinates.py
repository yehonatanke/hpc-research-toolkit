import torch
import logging


logger = logging.getLogger(__name__)


def convert_points_to_homogeneous(points: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D points to homogeneous coordinates by appending 1.
    """
    logger.debug("points", extra={"points": points.shape})
    logger.debug("ones", extra={"ones": torch.ones_like(points[..., :1]).shape})
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def convert_vectors_to_homogeneous(vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert 3D vectors to homogeneous coordinates by appending 0.
    """
    logger.debug("vectors", extra={"vectors": vectors.shape})
    logger.debug("zeros", extra={"zeros": torch.zeros_like(vectors[..., :1]).shape})
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def normalize_quaternion_sign(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Normalize quaternion sign by ensuring the scalar (w) component is non-negative.
    """
    logger.debug("quaternions", extra={"quaternions": quaternions.shape})
    logger.debug("where", extra={"where": torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions).shape})
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)
