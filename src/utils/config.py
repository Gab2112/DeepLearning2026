"""
Configuration loader and path resolver.
Reads YAML configs and automatically binds relative paths to PROJECT_ROOT,
ensuring execution works seamlessly across any IDE, working directory, or terminal.
"""

from pathlib import Path
from typing import Dict, Any, Union
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML config and resolves paths in the 'paths' section relative to PROJECT_ROOT.
    """
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Automatically resolve paths to absolute paths under PROJECT_ROOT
    if "paths" in cfg:
        for k, v in cfg["paths"].items():
            p = Path(v)
            if not p.is_absolute():
                cfg["paths"][k] = PROJECT_ROOT / p
            else:
                cfg["paths"][k] = p

    cfg["project_root"] = PROJECT_ROOT
    return cfg
