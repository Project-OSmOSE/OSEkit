from OSmOSE import utils
from OSmOSE.Auxiliary import Auxiliary
from OSmOSE.Dataset import Dataset
from OSmOSE.job import Job_builder
from OSmOSE.Spectrogram import Spectrogram
import OSmOSE.utils as utils
from OSmOSE.Auxiliary import Auxiliary
import logging.config
import yaml
import os.path
from pathlib import Path

__all__ = [
    "Auxiliary",
    "Dataset",
    "Job_builder",
    "Spectrogram",
    "utils",
]


def _setup_logging(config_file="logging_config.yaml", default_level=logging.INFO):

    user_config_file_path = Path(os.getenv("OSMOSE_USER_CONFIG", ".")) / config_file
    default_config_file_path = Path(os.path.dirname(__file__)) / config_file

    config_file_path = next(
        (
            file
            for file in (user_config_file_path, default_config_file_path)
            if file.exists()
        ),
        None,
    )

    if config_file_path:
        with open(config_file_path, "r") as config_file:
            logging_config = yaml.safe_load(config_file)
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=default_level)


_setup_logging()
