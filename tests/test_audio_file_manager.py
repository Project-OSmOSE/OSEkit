from __future__ import annotations

from OSmOSE.data.audio_file_manager import AudioFileManager
from OSmOSE.utils.audio_utils import generate_sample_audio
from pathlib import Path
import pytest
import numpy as np
from soundfile import LibsndfileError


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


@pytest.mark.parametrize(
    ("audio_files", "frames", "expected"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (-10, None),
            pytest.raises(
                ValueError,
                match="Start should be between 0 and the last frame of the audio file.",
            ),
            id="negative_start_raises_error",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (50_000, None),
            pytest.raises(
                ValueError,
                match="Start should be between 0 and the last frame of the audio file.",
            ),
            id="out_of_bounds_start_raises_error",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (None, -10),
            pytest.raises(
                ValueError,
                match="Stop should be between 0 and the last frame of the audio file.",
            ),
            id="negative_stop_raises_error",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (None, 50_000),
            pytest.raises(
                ValueError,
                match="Stop should be between 0 and the last frame of the audio file.",
            ),
            id="out_of_bounds_stop_raises_error",
        ),
    ],
    indirect=["audio_files"],
)
def test_read_errors(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    frames: tuple[int, int],
    expected: np.ndarray,
) -> None:
    audio_file_path = tmp_path / "audio_000101000000000000.wav"
    afm = AudioFileManager()
    params = {"start": frames[0], "stop": frames[1]}
    params = {k: v for k, v in params.items() if v is not None}
    with expected as e:
        assert afm.read(path=audio_file_path, **params) == e
