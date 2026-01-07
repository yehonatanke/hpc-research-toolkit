import numpy as np
import torch
import logging


logger = logging.getLogger(__name__)


def convert_to_homogeneous_matrix(ext):
    logger.debug("ext", extra={"ext": ext.shape})
    if isinstance(ext, torch.Tensor):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = torch.zeros_like(ext[..., :1, :4])
            logger.debug("ones", extra={"ones": ones.shape})
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            logger.error("Invalid shape for torch.Tensor", extra={"shape": ext.shape})
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")

    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            logger.error("Invalid shape for np.ndarray", extra={"shape": ext.shape})
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")

    else:
        logger.error("Input must be a torch.Tensor or np.ndarray.", extra={"input": type(ext)})
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


def compute_affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def swap_last_two_axes(arr):
    if arr.ndim < 2:
        logger.error("Array has less than 2 dimensions", extra={"arr": arr.shape})
        return arr
    axes = list(range(arr.ndim))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    logger.debug("Axes", extra={"axes": axes})
    return arr.transpose(axes)


def compute_affine_inverse_numpy(A: np.ndarray):
    R = A[..., :3, :3]
    logger.debug("R", extra={"R": R.shape})
    T = A[..., :3, 3:]
    logger.debug("T", extra={"T": T.shape})
    P = A[..., 3:, :]
    logger.debug("P", extra={"P": P.shape})
    return np.concatenate(
        [
            np.concatenate([swap_last_two_axes(R), -swap_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )
