import logging
import stat

from OSmOSE.logging_context import LoggingContext

TIMESTAMP_FORMAT_AUDIO_FILE = "%Y-%m-%dT%H:%M:%S.%f%z"
TIMESTAMP_FORMAT_TEST_FILES = "%y%m%d%H%M%S%f"
TIMESTAMP_FORMAT_EXPORTED_FILES = "%Y_%m_%d_%H_%M_%S_%f"
TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ = "%Y_%m_%d_%H_%M_%S_%f%z"
FPDEFAULT = 0o664  # Default file permissions
DPDEFAULT = stat.S_ISGID | 0o775  # Default directory permissions

global_logging_context = LoggingContext()
print_logger = logging.getLogger("printer")

resample_quality_settings = {
    "downsample": "QQ",
    "upsample": "MQ",
}
