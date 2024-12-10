import logging
import os
import stat
from collections import namedtuple
from pathlib import Path
from typing import TypeAlias

import soundfile as sf

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

FORBIDDEN_FILENAME_CHARACTERS = {":": "_", "-": "_"}
AUDIO_METADATA = {
    "filename": lambda f: f.name,
    "duration": lambda f: sf.info(f).duration,
    "origin_sr": lambda f: sf.info(f).samplerate,
    "sampwidth": lambda f: sf.info(f).subtype,
    "size": lambda f: f.stat().st_size / 1e6,
    "channel_count": lambda f: sf.info(f).channels,
}

global_logging_context = LoggingContext()
print_logger = logging.getLogger("printer")

BUILD_DURATION_DELTA_THRESHOLD = 0.05

FileName: TypeAlias = str | bytes | os.PathLike
