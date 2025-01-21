from __future__ import annotations

from OSmOSE.data.audio_file_manager import AudioFileManager
from OSmOSE.utils.audio_utils import generate_sample_audio
from pathlib import Path
import pytest
import numpy as np


@pytest.mark.parametrize(
    ("audio_files", "frames", "expected"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (None, None),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0],
            id="full_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (None, 10),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][:10],
            id="begin_of_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (10, None),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][10:],
            id="end_of_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (10, 10),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][10:10],
            id="mid_of_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_read(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    frames: tuple[int, int],
    expected: np.ndarray,
) -> None:

    audio_file_path = tmp_path / "audio_000101000000000000.wav"
    afm = AudioFileManager()
    params = {"start": frames[0], "stop": frames[1]}
    params = {k: v for k, v in params.items() if v is not None}
    assert np.array_equal(afm.read(path=audio_file_path, **params), expected)
