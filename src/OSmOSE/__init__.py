from pathlib import Path
from collections import namedtuple

from OSmOSE.Dataset import Dataset
from OSmOSE.timestamps import write_timestamp
from OSmOSE.job import Job_builder
from OSmOSE.Spectrogram import Spectrogram

__all__ = ["Dataset", "write_timestamp", "Job_builder", "Spectrogram", "utils"]

supported_audio_files = [".wav"]

_osmose_path_dict = {
    "raw_audio": Path("data", "audio"),
    "auxiliary": Path("data", "auxiliary"),
    "instrument": Path("data", "auxiliary", "instrument"),
    "environment": Path("data", "auxiliary", "instrument"),
    "processed": Path("processed"),
    "spectrograms": Path("processed", "spectrogram"),
}

_osmose_path_nt = namedtuple("path_list", _osmose_path_dict.keys())(**_osmose_path_dict)
