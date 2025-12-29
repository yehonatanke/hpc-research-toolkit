import logging

logger = logging.getLogger(__name__)

def subsample_plot_paths_list(result: dict, every_x: int = 1) -> dict:
    """
    Subsamples already extracted plot paths list, keeping every x-th frame.
    args:
        - result: dict, dictionary with keys: 'rgb_paths', 'frame_names', 'depth_paths'
        - every_x: int, every x-th element to keep
    returns:
        - dict, dictionary with keys: 'rgb_paths', 'frame_names', 'depth_paths' with every x-th element
    """
    if every_x <= 1:
        logger.warning("every_x is less than or equal to 1, returning original result")
        return result.copy()  
    
    subsampled = {
        "rgb_paths":   result["rgb_paths"][::every_x],
        "frame_names": result["frame_names"][::every_x],
        "depth_paths": result["depth_paths"][::every_x]
    }
    
    return subsampled

def subsample_paths(data: dict | tuple | list, every_x: int = 1):
    """
    keeps every x-th element from dict values, tuple items, or list items.
    args: 
        - data: dict, tuple, or list
        - every_x: int, every x-th element to keep
    returns:
        - dict, tuple, or list with every x-th element
    """
    if every_x <= 1:
        logger.warning("every_x is less than or equal to 1, returning original data")
        return data.copy() if isinstance(data, dict) else data[:]  
    
    if isinstance(data, dict):
        logger.info("data is a dict, subsampling each value")
        return {k: v[::every_x] for k, v in data.items()}
    
    if isinstance(data, (tuple, list)):
        logger.info("data is a tuple or list, subsampling each item")
        return type(data)(item[::every_x] for item in data)