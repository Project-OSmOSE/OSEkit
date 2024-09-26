from OSmOSE.config import SUPPORTED_AUDIO_FORMAT
from pathlib import Path


def is_audio(filename: Path) -> bool:

    return filename.suffix in SUPPORTED_AUDIO_FORMAT


def get_audio_file(file_path: Path) -> list[Path]:

    audio_path = []
    for ext in SUPPORTED_AUDIO_FORMAT:
        audio_path.extend(list(file_path.glob(f"*{ext}")))

    return sorted(audio_path)
