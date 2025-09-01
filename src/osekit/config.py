import logging
import stat

from osekit.logging_context import LoggingContext

TIMESTAMP_FORMAT_AUDIO_FILE = "%Y-%m-%dT%H:%M:%S.%f%z"
TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED = "%Y_%m_%d_%H_%M_%S_%f"
TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED = "%Y_%m_%d_%H_%M_%S_%f%z"
TIMESTAMP_FORMATS_EXPORTED_FILES = [
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
]

FPDEFAULT = 0o664  # Default file permissions
DPDEFAULT = stat.S_ISGID | 0o775  # Default directory permissions

global_logging_context = LoggingContext()
print_logger = logging.getLogger("printer")

resample_quality_settings = {
    "downsample": "QQ",
    "upsample": "MQ",
}

multiprocessing = {
    "is_active": False,
    "nb_processes": None,
}
