from __future__ import annotations

import logging.config
import os.path
from pathlib import Path

import yaml

from osekit import utils

__all__ = [
    "setup_logging",
    "utils",
]


def setup_logging(
    config_file: str | Path = "logging_config.yaml",
    default_level: int = logging.INFO,
) -> None:
    """Configure logger using a configuration yaml file.

    Parameters
    ----------
    config_file: str | Path
        Path to a logging configuration file
    default_level: int
        Logging level to use
        Default value, `logging.INFO`

    """
    user_config_file_path = Path(os.getenv("OSMOSE_USER_CONFIG", ".")) / config_file
    default_config_file_path = Path(__file__).parent / config_file

    config_file_path = next(
        (
            file
            for file in (user_config_file_path, default_config_file_path)
            if file.exists()
        ),
        None,
    )

    if config_file_path:
        with Path.open(config_file_path) as configuration:
            logging_config = yaml.safe_load(configuration)
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=default_level)
        msg = "Configuration file not found, using default configuration."
        logging.getLogger(__name__).warning(msg)
