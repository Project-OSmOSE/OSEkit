import os
import stat
from collections import namedtuple
from pathlib import Path
from typing import TypeAlias

from OSmOSE.logging_context import LoggingContext

SUPPORTED_AUDIO_FORMAT = [".wav", ".flac"]

# Dict are easier to modify, namedtuple easier to use
__global_path_dict = {
    "raw_audio": Path("data", "audio"),
    "auxiliary": Path("data", "auxiliary"),
    "instrument": Path("data", "auxiliary", "instrument"),
    "environment": Path("data", "auxiliary", "environment"),
    "processed": Path("processed"),
    "spectrogram": Path("processed", "spectrogram"),
    "statistics": Path("processed", "dataset_statistics"),
    "LTAS": Path("processed", "LTAS"),
    "welch": Path("processed", "welch"),
    "EPD": Path("processed", "EPD"),
    "SPLfiltered": Path("processed", "SPLfiltered"),
    "processed_auxiliary": Path("processed", "auxiliary"),
    "aplose": Path("processed", "aplose"),
    "weather": Path("appli", "weather"),
    "other": Path("other"),
    "log": Path("log"),
}

OSMOSE_PATH = namedtuple("path_list", __global_path_dict.keys())(**__global_path_dict)

TIMESTAMP_FORMAT_AUDIO_FILE = "%Y-%m-%dT%H:%M:%S.%f%z"
FPDEFAULT = 0o664  # Default file permissions
DPDEFAULT = stat.S_ISGID | 0o775  # Default directory permissions

global_logging_context = LoggingContext()

FileName: TypeAlias = str | bytes | os.PathLike
