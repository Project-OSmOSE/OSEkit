from OSmOSE.config import SUPPORTED_AUDIO_FORMAT
from pathlib import Path
import glob
import os


def is_audio(filename: Path):
    return any([filename.endswith(ext) for ext in SUPPORTED_AUDIO_FORMAT])


def get_audio_file(file_path: str | Path) -> list[Path]:

    assert any(
        [isinstance(file_path, str), isinstance(file_path, Path)]
    ), "A Path or string must be provided"

    audio_path = []
    for ext in SUPPORTED_AUDIO_FORMAT:
        audio_path_ext = glob.glob(os.path.join(file_path, f"*{ext}"))
        [audio_path.append(Path(f)) for f in audio_path_ext]

    return sorted(audio_path)
