import json
import os
from typing import Dict, Any, Optional


def load_config(
    config_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    config_filename: str = "default.json",
    default_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load default configuration from a JSON file.

    Args:
        - config_path: Direct path to the config file. If provided, this takes precedence.
        - base_dir: Base directory to search for config file. If not provided, will try to
            auto-detect from the calling file's location.
        - config_filename: Name of the config file (default: "default.json").
        - default_values: Dictionary of default values to use if config file is not found
            or cannot be loaded.

    Returns:
        - Dictionary containing the loaded configuration values, merged with default_values.
    """
    if default_values is None:
        default_values = {}

    if config_path is None:
        if base_dir is None:
            import inspect

            try:
                frame = inspect.currentframe()
                if frame is not None:
                    caller_frame = frame.f_back
                    if caller_frame is not None:
                        caller_file = caller_frame.f_globals.get("__file__", "")
                        if caller_file:
                            # Go up from caller's directory to find config folder
                            caller_dir = os.path.dirname(os.path.abspath(caller_file))
                            # Look for config folder in parent directories (up to 3 levels)
                            for _ in range(3):
                                potential_config_dir = os.path.join(caller_dir, "configs")
                                potential_config_path = os.path.join(potential_config_dir, config_filename)
                                if os.path.exists(potential_config_path):
                                    config_path = potential_config_path
                                    break
                                caller_dir = os.path.dirname(caller_dir)
            except Exception:
                pass

        if config_path is None and base_dir is not None:
            # Use provided base_dir
            config_dir = os.path.join(base_dir, "configs")
            config_path = os.path.join(config_dir, config_filename)
        elif config_path is None:
            # Fallback: try common locations
            config_path = os.path.join("configs", config_filename)

    # Load config file if it exists
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                # Merge with defaults (loaded config takes precedence)
                result = {**default_values, **loaded_config}
                return result
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load default config from {config_path}: {e}")
            print("Using provided defaults.")
            return default_values.copy()
    else:
        # Config file doesn't exist, return defaults
        if config_path:
            print(f"Warning: Config file not found at {config_path}")
            print("Using provided defaults.")
        return default_values.copy()
