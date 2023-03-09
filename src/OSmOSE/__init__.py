from pathlib import Path
from collections import namedtuple

from OSmOSE.Dataset import Dataset
from OSmOSE.timestamps import write_timestamp
from OSmOSE.job import Job_builder
from OSmOSE.Spectrogram import Spectrogram

__all__ = ["Dataset", "write_timestamp", "Job_builder", "Spectrogram", "utils"]

supported_audio_files = [".wav"]

# Dict are easier to modify, namedtuple easier to use
__global_path_dict = {
    "raw_audio": Path("data", "audio"),
    "auxiliary": Path("data", "auxiliary"),
    "instrument": Path("data", "auxiliary", "instrument"),
    "environment": Path("data", "auxiliary", "instrument"),
    "processed": Path("processed"),
    "spectrogram": Path("processed", "spectrogram"),
}

_OSMOSE_PATH = namedtuple("path_list", __global_path_dict.keys())(**__global_path_dict)
