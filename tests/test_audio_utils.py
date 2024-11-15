from pathlib import Path

import pytest

from OSmOSE.utils.audio_utils import is_supported_audio_format


@pytest.mark.unit
@pytest.mark.parametrize(
    ("filepath", "expected_output"),
    [
        pytest.param(Path("audio.wav"), True, id="simple_wav_file"),
        pytest.param(Path("audio_with_date_2024_02_14.wav"), True, id="complex_wav_file"),
        pytest.param(Path("parent_folder/audio.wav"), True, id="file_in_parent_folder"),
        pytest.param(Path("audio.flac"), True, id="simple_flac_file"),
        pytest.param(Path("audio.WAV"), True, id="uppercase_wav_extension"),
        pytest.param(Path("audio.FLAC"), True, id="uppercase_flac_extension"),
        pytest.param(Path("audio.mp3"), False, id="unsupported_audio_extension"),
        pytest.param(Path("parent_folder/audio.MP3"), False, id="unsupported_in_parent_folder"),
        pytest.param(Path("audio.pdf"), False, id="unsupported_extension"),
        pytest.param(Path("audio"), False, id="no_extension"),
    ],
)
def test_supported_audio_formats(filepath: Path, expected_output: bool) -> None:
    assert is_supported_audio_format(filepath) == expected_output

