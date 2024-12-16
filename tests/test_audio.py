from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from OSmOSE.config import TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.data.audio_data import AudioData
from OSmOSE.data.audio_dataset import AudioDataset
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
        pytest.param(
            {
                "duration": 1,
                "inter_file_duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            np.array(
                list(
                    generate_sample_audio(nb_files=1, nb_samples=48_000 * 2)[0][
                        0:48_000
                    ],
                )
                + [0.0] * 48_000
                + list(
                    generate_sample_audio(nb_files=1, nb_samples=48_000 * 2)[0][
                        48_000:
                    ],
                ),
            ),
            id="empty_space_is_filled",
        ),
        pytest.param(
            {
                "duration": 1,
                "inter_file_duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            pd.Timestamp("2024-01-01 10:00:00"),
            pd.Timestamp("2024-01-01 10:00:01"),
            np.zeros(shape=48_000),
            id="out_of_range_is_zeros",
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
    if all(item.is_empty for item in data.items):
        data.sample_rate = 48_000
    assert np.array_equal(data.get_value(), expected)


@pytest.mark.parametrize(
    ("audio_files", "start", "stop", "sample_rate", "expected_nb_samples"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            24_000,
            24_000,
            id="downsampling",
        ),
        pytest.param(
            {
                "duration": 0.5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            96_000,
            48_000,
            id="upsampling",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            pd.Timestamp("2024-01-01 12:00:01"),
            None,
            96_000,
            96_000,
            id="upsampling_file_part",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            16_000,
            48_000,
            id="downsampling_with_gaps",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_resample_sample_count(
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    start: pd.Timestamp | None,
    stop: pd.Timestamp | None,
    sample_rate: int,
    expected_nb_samples: int,
) -> None:
    files, request = audio_files
    audio_files = [
        AudioFile(file, strptime_format=TIMESTAMP_FORMAT_TEST_FILES) for file in files
    ]
    data = AudioData.from_files(audio_files, begin=start, end=stop)
    data.sample_rate = sample_rate
    assert data.get_value().shape[0] == expected_nb_samples


@pytest.mark.parametrize(
    ("audio_files", "begin", "end", "duration", "expected_audio_data"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            None,
            generate_sample_audio(1, 48_000),
            id="one_entire_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            pd.Timedelta(seconds=1),
            generate_sample_audio(
                nb_files=3, nb_samples=48_000, series_type="increase"
            ),
            id="multiple_consecutive_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            pd.Timedelta(seconds=1),
            [
                generate_sample_audio(nb_files=1, nb_samples=96_000)[0][0:48_000],
                generate_sample_audio(
                    nb_files=1, nb_samples=48_000, min_value=0.0, max_value=0.0
                )[0],
                generate_sample_audio(nb_files=1, nb_samples=96_000)[0][48_000:],
            ],
            id="two_separated_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "inter_file_duration": -0.5,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "repeat",
            },
            None,
            None,
            pd.Timedelta(seconds=1),
            generate_sample_audio(nb_files=2, nb_samples=48_000),
            id="overlapping_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_dataset_from_folder(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    begin: pd.Timestamp | None,
    end: pd.Timestamp | None,
    duration: pd.Timedelta | None,
    expected_audio_data: list[tuple[int, bool]],
) -> None:
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
        begin=begin,
        end=end,
        data_duration=duration,
    )
    assert all(
        np.array_equal(data.get_value(), expected)
        for (data, expected) in zip(dataset.data, expected_audio_data)
    )
