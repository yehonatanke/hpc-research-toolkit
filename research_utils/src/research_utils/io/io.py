import json
import os
import yaml
import inspect
from typing import Dict, Any, Optional, List


def _expand_env_vars_v1(obj):
    """
    Recursively expand environment variables in the object.
    Only processes strings; dicts and lists are traversed recursively.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_vars_v1(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars_v1(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    else:
        return obj


def _expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    elif isinstance(obj, str):
        return os.path.expandvars(obj)
    return obj


def _get_caller_directory() -> Optional[str]:
    try:
        frame = inspect.currentframe()
        if frame and frame.f_back and frame.f_back.f_back:
            caller_file = frame.f_back.f_back.f_globals.get("__file__", "")
            if caller_file:
                return os.path.dirname(os.path.abspath(caller_file))
    except Exception:
        pass
    return None


def _find_config_in_parents(start_dir: str, filenames: List[str], depth: int = 3) -> Optional[str]:
    current_dir = start_dir
    for _ in range(depth):
        config_dir = os.path.join(current_dir, "configs")
        for name in filenames:
            path = os.path.join(config_dir, name)
            if os.path.exists(path):
                return path
        current_dir = os.path.dirname(current_dir)
    return None


def _resolve_config_path(config_path: Optional[str], base_dir: Optional[str], search_names: List[str]) -> Optional[str]:
    if config_path:
        return config_path

    # Search from caller directory upwards
    caller_dir = _get_caller_directory()
    if caller_dir:
        found = _find_config_in_parents(caller_dir, search_names)
        if found:
            return found

    # Search in provided base_dir or local 'configs'
    search_root = base_dir if base_dir else os.getcwd()
    config_dir = os.path.join(search_root, "configs")
    for name in search_names:
        path = os.path.join(config_dir, name)
        if os.path.exists(path):
            return path

    return None


def _parse_file(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        if path.lower().endswith((".yaml", ".yml")):
            return yaml.safe_load(f) or {}
        return json.load(f) or {}


def load_config(
    config_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    config_filename: Optional[str] = None,
    default_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    defaults = default_values or {}
    search_names = [config_filename] if config_filename else ["default.yaml", "default.yml", "default.json"]

    resolved_path = _resolve_config_path(config_path, base_dir, search_names)

    if resolved_path and os.path.exists(resolved_path):
        try:
            loaded_config = _parse_file(resolved_path)
            return _expand_env_vars({**defaults, **loaded_config})
        except (json.JSONDecodeError, yaml.YAMLError, IOError) as e:
            print(f"Warning: Failed to load {resolved_path}: {e}")

    return _expand_env_vars(defaults.copy())


def load_config_json(
    config_path: Optional[str] = None,
    base_dir: Optional[str] = None,
    config_filename: str = "default.json",
    default_values: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load default configuration from a JSON file.

    This function will expand environment variables in any string value in the loaded config.
    For example, if your JSON contains {"data_dir": "$WORK/data"}, and $WORK is set in your environment,
    the returned dictionary will have {"data_dir": "/expanded/path/data"}.

    Args:
        - config_path: Direct path to the config file. If provided, this takes precedence.
        - base_dir: Base directory to search for config file. If not provided, will try to
            auto-detect from the calling file's location.
        - config_filename: Name of the config file (default: "default.json").
        - default_values: Dictionary of default values to use if config file is not found
            or cannot be loaded.

    Returns:
        - Dictionary containing the loaded configuration values, merged with default_values,
          and with environment variables expanded in any string values.
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
                # Expand environment variables in any string fields
                result = _expand_env_vars(result)
                return result
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load default config from {config_path}: {e}")
            print("Using provided defaults.")
            return _expand_env_vars(default_values.copy())
    else:
        # Config file doesn't exist, return defaults
        if config_path:
            print(f"Warning: Config file not found at {config_path}")
            print("Using provided defaults.")
        return _expand_env_vars(default_values.copy())
