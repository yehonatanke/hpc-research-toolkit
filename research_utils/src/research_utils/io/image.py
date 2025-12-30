import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


def read_rgb(rgb_path):
    """
    read rgb image
    returns:
        - rgb_img: rgb image
    """
    rgb_img = cv2.imread(rgb_path)
    if rgb_img is None:
        logger.error(f"Error reading RGB: {rgb_path}.")
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    return rgb_img


def get_rgb_image(rgb_path: str):
    img = cv2.imread(rgb_path)
    if img is None:
        logger.error("RGB file not found", extra={"path": rgb_path})
        raise FileNotFoundError(f"Error reading RGB: {rgb_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_depth_map(depth_path: str):
    if depth_path.endswith(".npy"):
        depth_map = np.load(depth_path)
        logger.debug("Depth file is a numpy array", extra={"path": depth_path})
    else:
        depth_map = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        logger.debug("Depth file is not a numpy array", extra={"path": depth_path})

    if depth_map is None:
        logger.error("Depth file not found", extra={"path": depth_path})
        raise FileNotFoundError(f"Error reading Depth: {depth_path}")

    return depth_map


def get_image_dimensions(image_path: str):
    """
    Get image dimensions (width, height) from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple of (width, height) in pixels, or (None, None) if image cannot be read
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is not None:
        height, width = img.shape[:2]
        logger.debug("Image dimensions detected", extra={"path": image_path, "width": width, "height": height})
        return width, height
    logger.warning("Could not read image to get dimensions", extra={"path": image_path})
    return None, None
