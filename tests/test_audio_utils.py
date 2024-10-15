import pytest
from pathlib import Path
from OSmOSE.utils.audio_utils import *

@pytest.mark.unit
@pytest.mark.parametrize('filepath, expected_output', [
    (Path("audio.wav"), True),
    (Path("audio_with_date_2024_02_14.wav"), True),
    (Path("parent_folder/audio.wav"), True),
    (Path("audio.flac"), True),
    (Path("audio.WAV"), True),
    (Path("audio.FLAC"), True),
    (Path("audio.mp3"), False),
    (Path("audio.MP3"), False),
    (Path("audio.pdf"), False),
])
def test_supported_audio_formats(filepath: Path, expected_output: bool):
    assert is_supported_audio_format(filepath) == expected_output

