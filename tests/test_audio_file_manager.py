from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from osekit.audio_backend.audio_file_manager import AudioFileManager
from osekit.utils.audio_utils import generate_sample_audio

if TYPE_CHECKING:
    from osekit.core_api.audio_file import AudioFile


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
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    frames: tuple[int, int],
    expected: np.ndarray,
) -> None:
    audio_files, _ = audio_files
    afm = AudioFileManager()
    params = {"start": frames[0], "stop": frames[1]}
    params = {k: v for k, v in params.items() if v is not None}
    assert np.array_equal(afm.read(path=audio_files[0].path, **params), expected)


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
                match=r"Start should be between 0 and the last frame of the audio file.",
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
                match=r"Start should be between 0 and the last frame of the audio file.",
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
                match=r"Stop should be between 0 and the last frame of the audio file.",
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
                match=r"Stop should be between 0 and the last frame of the audio file.",
            ),
            id="out_of_bounds_stop_raises_error",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            (20_000, 10_000),
            pytest.raises(
                ValueError,
                match=r"Start should be inferior to Stop.",
            ),
            id="start_after_stop_raises_error",
        ),
    ],
    indirect=["audio_files"],
)
def test_read_errors(
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    frames: tuple[int, int],
    expected: np.ndarray,
) -> None:
    audio_files, _ = audio_files
    afm = AudioFileManager()
    params = {"start": frames[0], "stop": frames[1]}
    params = {k: v for k, v in params.items() if v is not None}
    with expected as e:
        assert afm.read(path=audio_files[0].path, **params) == e


@pytest.mark.parametrize(
    ("audio_files", "file_openings", "expected_opened_files"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
            },
            [0],
            [0],
            id="one_single_file_opening",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
            },
            [0, 0, 0, 0, 0],
            [0],
            id="repeated_file_openings",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 5,
            },
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            id="different_file_openings",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 5,
            },
            [0, 0, 0, 1, 1, 1, 1, 2, 3, 3, 4, 4, 4, 2, 2, 1, 1],
            [0, 1, 2, 3, 4, 2, 1],
            id="multiple_repeated_file_openings",
        ),
    ],
    indirect=["audio_files"],
)
def test_switch(
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    file_openings: list[int],
    patch_afm_open: list[Path],
    expected_opened_files: list[int],
) -> None:
    afm = AudioFileManager()
    sf_back = afm._soundfile
    audio_files, _ = audio_files
    audio_files = [af.path for af in audio_files]
    for file in file_openings:
        afm.read(path=audio_files[file])
    assert [audio_files.index(f) for f in patch_afm_open] == expected_opened_files
    assert audio_files.index(Path(sf_back._file.name)) == file_openings[-1]


@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
            },
            id="one_single_file_opening",
        ),
    ],
    indirect=True,
)
def test_close(audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest]) -> None:
    afm = AudioFileManager()
    sf_back = afm._soundfile
    assert sf_back._file is None
    audio_files, _ = audio_files
    afm.read(audio_files[0].path)
    assert sf_back._file is not None
    assert Path(sf_back._file.name) == audio_files[0].path
    afm.close()
    assert sf_back._file is None


@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 5,
            },
            id="multiple_files",
        ),
    ],
    indirect=True,
)
def test_info(audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest]) -> None:
    afm = AudioFileManager()
    sf_back = afm._soundfile
    audio_files, request = audio_files
    for file in audio_files:
        assert sf_back._file is None or Path(sf_back._file.name) != file.path.name
        sample_rate, frames, channels = afm.info(file.path)
        assert request.param["sample_rate"] == sample_rate
        assert request.param["duration"] * request.param["sample_rate"] == frames
        assert channels == 1
