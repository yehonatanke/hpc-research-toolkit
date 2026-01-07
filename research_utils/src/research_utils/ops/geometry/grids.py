import torch
import logging


logger = logging.getLogger(__name__)


def generate_image_coordinate_grid(
    shape: tuple[int, ...], device: torch.device = torch.device("cpu")
) -> tuple[torch.Tensor, torch.Tensor,]:
    logger.debug("shape", extra={"shape": shape})
    logger.debug("device", extra={"device": device})
    indices = [torch.arange(length, device=device) for length in shape]
    
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)
    logger.debug("coordinates", extra={"coordinates": coordinates.shape})
    logger.debug("stacked_indices", extra={"stacked_indices": stacked_indices.shape})
    
    return coordinates, stacked_indices
