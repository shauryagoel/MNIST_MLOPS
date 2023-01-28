# Hold some basic utility functions
from pathlib import Path

import yaml


def load_yaml_config(file_path: Path):
    """Load the yaml config file."""
    with open(file_path, "r") as f:
        config = yaml.load(f, yaml.Loader)
    return config
