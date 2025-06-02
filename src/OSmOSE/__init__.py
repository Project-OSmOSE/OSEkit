import logging.config
import os.path
from pathlib import Path

import yaml

from OSmOSE import utils
from OSmOSE.job import Job_builder

__all__ = [
    "Job_builder",
    "utils",
]


def _setup_logging(
    config_file="logging_config.yaml",
    default_level: int = logging.INFO,
) -> None:
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


_setup_logging()
