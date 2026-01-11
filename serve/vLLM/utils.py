"""
Utility functions for vLLM server configuration
"""
import yaml
import os
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    """
    Load configuration from config.yaml file.

    Args:
        config_path: Optional path to config file. If not provided, looks for config.yaml
                    in the same directory as this script.

    Returns:
        Dictionary containing configuration
    """
    if config_path is None:
        # Get the directory where this script is located
        script_dir = Path(__file__).parent.absolute()
        config_path = script_dir / "config.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config
