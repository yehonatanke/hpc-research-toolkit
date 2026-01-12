"""
Test that load_config and load_config_json return the same results for JSON files.

This test verifies that both functions produce identical outputs when loading
the same JSON configuration file, ensuring consistency across the codebase.
"""

import json
import os
import tempfile
from pathlib import Path

from research_utils.io.io import load_config, load_config_json

# optional override: use TEST_JSON_CONFIG_PATH="..." to check a specific JSON file.
TEST_JSON_PATH = os.environ.get("TEST_JSON_CONFIG_PATH", "").strip()


def test_json_equivalence():
    """Test that load_config and load_config_json return the same results for JSON files."""
    # Test data cases
    test_cases = [
        {"name": "Simple key-value pairs", "data": {"key1": "value1", "key2": 42, "key3": True}},
        {"name": "Nested dictionaries", "data": {"level1": {"level2": {"level3": "deep_value"}}}},
        {"name": "Lists and mixed types", "data": {"list": [1, 2, 3], "mixed": {"a": 1, "b": [4, 5, 6]}}},
        {"name": "Environment variables", "data": {"path": "$HOME/test", "normal": "text", "number": 123}},
        {"name": "Empty dict", "data": {}},
        {
            "name": "Complex nested structure",
            "data": {
                "config": {
                    "database": {"host": "localhost", "port": 5432},
                    "features": ["feature1", "feature2"],
                    "enabled": True,
                }
            },
        },
    ]

    for test_case in test_cases:
        test_data = test_case["data"]

        # Use override path if provided, otherwise create a temp JSON file
        if TEST_JSON_PATH:
            json_path = TEST_JSON_PATH
            cleanup = False
        else:
            cleanup = True
            # Create temporary JSON file
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
                json.dump(test_data, f)
                json_path = f.name

        try:
            # Test both functions with explicit path
            result1 = load_config(config_path=json_path)
            result2 = load_config_json(config_path=json_path)

            # Results should be identical
            assert result1 == result2, (
                f"Results differ for test case '{test_case['name']}':\n"
                f"  load_config:      {result1}\n"
                f"  load_config_json: {result2}"
            )
        finally:
            # Clean up temp file only if we created it
            if cleanup:
                Path(json_path).unlink()


def test_json_equivalence_with_defaults():
    """Test that both functions handle default values the same way for JSON files."""
    test_data = {"key1": "value1", "key2": 42}
    defaults = {"key2": 100, "key3": "default_value"}

    cleanup = not bool(TEST_JSON_PATH)

    if TEST_JSON_PATH:
        json_path = TEST_JSON_PATH
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name

    try:
        result1 = load_config(config_path=json_path, default_values=defaults)
        result2 = load_config_json(config_path=json_path, default_values=defaults)

        assert result1 == result2, (
            f"Results differ with defaults:\n" f"  load_config:      {result1}\n" f"  load_config_json: {result2}"
        )
    finally:
        if cleanup:
            Path(json_path).unlink()


def test_env_var_expansion_equivalence():
    """Test that both functions expand environment variables the same way."""
    # Set a test environment variable
    os.environ["TEST_VAR"] = "test_value"

    test_data = {"path1": "$TEST_VAR/subdir", "path2": "$HOME/test", "normal": "no_expansion"}

    cleanup = not bool(TEST_JSON_PATH)

    if TEST_JSON_PATH:
        json_path = TEST_JSON_PATH
    else:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            json_path = f.name

    try:
        result1 = load_config(config_path=json_path)
        result2 = load_config_json(config_path=json_path)

        assert result1 == result2, (
            f"Environment variable expansion differs:\n"
            f"  load_config:      {result1}\n"
            f"  load_config_json: {result2}"
        )
    finally:
        if cleanup:
            Path(json_path).unlink()
        # Clean up test env var
        os.environ.pop("TEST_VAR", None)


def test_yaml_vs_json_with_load_config():
    """Test that load_config returns the same result for YAML and JSON files with same content."""
    import yaml

    test_data = {"key1": "value1", "key2": 42, "key3": [1, 2, 3], "key4": {"nested": "value"}, "env_var": "$HOME/test"}

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create configs directory
        configs_dir = Path(tmpdir) / "configs"
        configs_dir.mkdir()

        # Create JSON file
        json_path = configs_dir / "default.json"
        with open(json_path, "w") as f:
            json.dump(test_data, f)

        # Create YAML file with same content
        yaml_path = configs_dir / "default.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(test_data, f)

        # Test load_config with both formats
        result_json = load_config(config_path=str(json_path))
        result_yaml = load_config(config_path=str(yaml_path))

        # Results should be the same
        assert result_json == result_yaml, (
            f"YAML and JSON results differ:\n" f"  JSON result: {result_json}\n" f"  YAML result: {result_yaml}"
        )
