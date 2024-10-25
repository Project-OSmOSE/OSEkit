from OSmOSE.Dataset import Dataset
from OSmOSE.job import Job_builder
from OSmOSE.Spectrogram import Spectrogram
import OSmOSE.utils as utils
from OSmOSE.Auxiliary import Auxiliary
import logging.config
import yaml
import os.path

__all__ = [
    "Auxiliary",
    "Dataset",
    "Job_builder",
    "Spectrogram",
    "utils",
]

def _setup_logging(config_file = "logging_conf.yaml", default_level= logging.INFO):
    config_file_path = os.path.join(os.path.dirname(__file__), config_file)
    if os.path.exists(config_file_path):
        with open(config_file_path, "r") as config_file:
            logging_config = yaml.safe_load(config_file)
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=default_level)

_setup_logging()