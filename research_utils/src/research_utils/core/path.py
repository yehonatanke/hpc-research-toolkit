import os


def path_with_parents(pathname: str, num_levels: int = 2) -> str:
    """
    Returns 'parent2/parent1/filename' (num_levels parents before file) for the given path.
    
    Args:
        pathname: The full path to process
        num_levels: Number of parent directories to include (default: 2)
    
    Returns:
        A relative path string with the specified number of parent directories and the filename.
        Returns empty string if pathname is empty.
    
    Examples:
        >>> path_with_parents("/a/b/c/d/file.py", 2)
        'c/d/file.py'
        >>> path_with_parents("/a/b/file.py", 2)
        'b/file.py'
        >>> path_with_parents("file.py", 2)
        'file.py'
    """
    if not pathname:
        return ""
    parts = os.path.normpath(pathname).split(os.sep)
    if len(parts) >= (num_levels + 1):
        rel_path = os.path.join(*parts[-(num_levels+1):])
    elif len(parts) >= 2:
        rel_path = os.path.join(parts[-2], parts[-1])
    else:
        rel_path = parts[-1]
    return rel_path

