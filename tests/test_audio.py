from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from OSmOSE.config import TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.data.audio_data import AudioData
from OSmOSE.data.audio_file import AudioFile
from OSmOSE.data.audio_item import AudioItem
from OSmOSE.utils.audio_utils import generate_sample_audio

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 0.05,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            id="basic_audio_file",
        ),
        pytest.param(
            {
                "duration": 0.06,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            id="different_duration",
        ),
        pytest.param(
            {
                "duration": 0.05,
                "sample_rate": 44_100,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            id="different_sample_rate",
        ),
    ],
    indirect=True,
)
def test_audio_file_timestamps(
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
) -> None:
    files, request = audio_files
    duration = request.param["duration"]
    date_begin = request.param["date_begin"]

    for file in files:
        audio_file = AudioFile(file, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)

        assert audio_file.begin == date_begin
        assert audio_file.end == date_begin + pd.Timedelta(seconds=duration)


@pytest.mark.parametrize(
    ("audio_files", "start", "stop", "expected"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            pd.Timestamp("2024-01-01 12:00:00"),
            pd.Timestamp("2024-01-01 12:00:01"),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0],
            id="read_whole_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            pd.Timestamp("2024-01-01 12:00:00"),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=100_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][:4_800],
            id="read_begin_only",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=500_000,
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=600_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][24_000:28_800],
            id="read_in_the_middle_of_the_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=900_000,
            ),
            pd.Timestamp("2024-01-01 12:00:01"),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][43_200:],
            id="read_end_of_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_file_read(
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    start: pd.Timestamp,
    stop: pd.Timestamp,
    expected: np.ndarray,
) -> None:
    files, request = audio_files
    file = AudioFile(files[0], strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
    assert np.array_equal(file.read(start, stop), expected)


@pytest.mark.parametrize(
    ("audio_files", "start", "stop", "expected"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0],
            id="whole_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=500_000,
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=600_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][24_000:28_800],
            id="mid_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_item(
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    start: pd.Timestamp | None,
    stop: pd.Timestamp | None,
    expected: np.ndarray,
) -> None:
    files, request = audio_files
    file = AudioFile(files[0], strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
    item = AudioItem(file, start, stop)
    assert np.array_equal(item.get_value(), expected)


@pytest.mark.parametrize(
    ("audio_files", "start", "stop", "expected"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            generate_sample_audio(nb_files=1, nb_samples=48_000 * 2)[0],
            id="all_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=800_000,
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=1,
                microsecond=200_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=48_000 * 2)[0][38_400:57_600],
            id="between_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_data(
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    start: pd.Timestamp | None,
    stop: pd.Timestamp | None,
    expected: np.ndarray,
) -> None:
    files, request = audio_files
    audio_files = [
        AudioFile(file, strptime_format=TIMESTAMP_FORMAT_TEST_FILES) for file in files
    ]
    data = AudioData.from_files(audio_files, begin=start, end=stop)
    assert np.array_equal(data.get_value(), expected)
