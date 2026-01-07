import torch
import torch.nn.functional as F
import logging

from .coordinates import normalize_quaternion_sign


logger = logging.getLogger(__name__)


def quaternion_to_rotation_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    logger.debug("quaternions", extra={"quaternions": quaternions.shape})
    i, j, k, r = torch.unbind(quaternions, -1)
    logger.debug("i", extra={"i": i.shape})
    logger.debug("j", extra={"j": j.shape})
    logger.debug("k", extra={"k": k.shape})
    logger.debug("r", extra={"r": r.shape})
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    logger.debug("two_s", extra={"two_s": two_s.shape})
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    logger.debug("o", extra={"o": o.shape})
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    logger.debug("matrix", extra={"matrix": matrix.shape})
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        logger.error("Invalid rotation matrix shape", extra={"shape": matrix.shape})
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    logger.debug("batch_dim", extra={"batch_dim": batch_dim})
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))
    out = out[..., [1, 2, 3, 0]]
    out = normalize_quaternion_sign(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    logger.debug("x", extra={"x": x.shape})
    ret = torch.zeros_like(x)
    logger.debug("ret", extra={"ret": ret.shape})
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret
