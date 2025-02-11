from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "format": "flac",
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
            id="flac_file",
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
    assert np.allclose(file.read(start, stop), expected, atol=1e-7)


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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
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
                microsecond=320_000,
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
            generate_sample_audio(nb_files=1, nb_samples=10)[0][3:6],
            id="start_between_frames",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
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
                microsecond=300_000,
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=620_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=10)[0][3:6],
            id="stop_between_frames",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
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
                microsecond=290_000,
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=790_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=10)[0][2:8],
            id="first_frame_included_last_frame_rounding_up",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
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
                microsecond=290_000,
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=720_000,
            ),
            generate_sample_audio(nb_files=1, nb_samples=10)[0][2:7],
            id="first_frame_included_last_frame_rounding_down",
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
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            id="simple_audio",
        ),
        pytest.param(
            {
                "duration": 14.303492063,
                "sample_rate": 44_100,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            id="uneven_boundaries_rounding_up",
        ),
        pytest.param(
            {
                "duration": 14.303471655328797,
                "sample_rate": 44_100,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            id="uneven_boundaries_rounding_down",
        ),
    ],
    indirect=True,
)
def test_read_vs_soundfile(
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
) -> None:
    audio_files, _ = audio_files
    af = AudioFile(audio_files[0], strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
    ad = AudioData.from_files([af])
    assert np.array_equal(sf.read(audio_files[0])[0], ad.get_value())


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
                nb_files=3,
                nb_samples=48_000,
                series_type="increase",
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
                    nb_files=1,
                    nb_samples=48_000,
                    min_value=0.0,
                    max_value=0.0,
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
    expected_audio_data: list[np.ndarray],
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


@pytest.mark.parametrize(
    (
        "audio_files",
        "expected_audio_data",
        "corrupted_audio_files",
        "non_audio_files",
        "error",
    ),
    [
        pytest.param(
            {"nb_files": 0},
            [],
            [],
            [],
            pytest.raises(
                FileNotFoundError,
                match="No valid audio file found in ",
            ),
            id="no_file",
        ),
        pytest.param(
            {"nb_files": 0},
            [],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".flac",
            ],
            [],
            pytest.raises(
                FileNotFoundError,
                match="No valid audio file found in ",
            ),
            id="corrupted_audio_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            generate_sample_audio(
                nb_files=1,
                nb_samples=144_000,
                series_type="increase",
            ),
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".flac",
            ],
            [],
            None,
            id="mixed_audio_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            generate_sample_audio(
                nb_files=1,
                nb_samples=144_000,
                series_type="increase",
            ),
            [],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".csv",
            ],
            None,
            id="non_audio_files_are_not_logged",
        ),
        pytest.param(
            {"nb_files": 0},
            [],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".flac",
            ],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".csv",
            ],
            pytest.raises(
                FileNotFoundError,
                match="No valid audio file found in ",
            ),
            id="all_but_ok_audio",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            generate_sample_audio(
                nb_files=1,
                nb_samples=144_000,
                series_type="increase",
            ),
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".flac",
            ],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_TEST_FILES,
                )
                + ".csv",
            ],
            None,
            id="full_mix",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_dataset_from_folder_errors_warnings(
    tmp_path: Path,
    caplog,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    expected_audio_data: list[np.ndarray],
    corrupted_audio_files: list[str],
    non_audio_files: list[str],
    error,
) -> None:

    for corrupted_file in corrupted_audio_files:
        (tmp_path / corrupted_file).open("a").close()  # Write empty audio files.

    with caplog.at_level(logging.WARNING):
        if error is not None:
            with error as e:
                assert (
                    AudioDataset.from_folder(
                        tmp_path,
                        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
                    )
                    == e
                )
        else:
            dataset = AudioDataset.from_folder(
                tmp_path,
                strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
            )
            assert all(
                np.array_equal(data.get_value(), expected)
                for (data, expected) in zip(dataset.data, expected_audio_data)
            )
        assert all(f in caplog.text for f in corrupted_audio_files)


@pytest.mark.parametrize(
    ("audio_files", "subtype", "expected_audio_data"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            "DOUBLE",
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="float64_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            "FLOAT",
            generate_sample_audio(1, 48_000, dtype=np.float32),
            id="float32_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            "DOUBLE",
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="padded_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_write_files(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    subtype: str,
    expected_audio_data: list[np.ndarray],
) -> None:
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
    )
    output_path = tmp_path / "output"
    dataset.write(output_path, subtype=subtype)
    for data in dataset.data:
        assert f"{data}.wav" in [f.name for f in output_path.glob("*.wav")]
        assert np.allclose(data.get_value(), sf.read(output_path / f"{data}.wav")[0])


@pytest.mark.parametrize(
    ("audio_files", "nb_subdata", "original_audio_data"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            2,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="even_samples_split_in_two",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            4,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="even_samples_split_in_four",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_001,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            2,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="odd_samples_split_in_two",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_001,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            4,
            generate_sample_audio(1, 48_001, dtype=np.float64),
            id="odd_samples_split_in_four",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            3,
            generate_sample_audio(1, 10, dtype=np.float64),
            id="infinite_decimal_points",
        ),
    ],
    indirect=["audio_files"],
)
def test_split_data(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    nb_subdata: int,
    original_audio_data: list[np.ndarray],
) -> None:
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
    )
    for data in dataset.data:
        subdata_shape = data.shape // nb_subdata
        for subdata, data_range in zip(
            data.split(nb_subdata),
            range(0, data.shape, subdata_shape),
        ):
            assert np.array_equal(
                subdata.get_value(),
                data.get_value()[data_range : data_range + subdata_shape],
            )


@pytest.mark.parametrize(
    ("audio_files", "start_frame", "stop_frame", "expected_begin", "expected_data"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            0,
            -1,
            pd.Timestamp("2024-01-01 12:00:00"),
            generate_sample_audio(1, 48_000, dtype=np.float64)[0],
            id="whole_data",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            0,
            1,
            pd.Timestamp("2024-01-01 12:00:00"),
            generate_sample_audio(1, 48_000, dtype=np.float64)[0][:1],
            id="first_frame",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            47_999,
            -1,
            pd.Timestamp("2024-01-01 12:00:00")
            + pd.Timedelta(seconds=round(47_999 / 48_000, 9)),
            generate_sample_audio(1, 48_000, dtype=np.float64)[0][-1:],
            id="last_frame",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            4_800 * 3,
            4_800 * 4,
            pd.Timestamp("2024-01-01 12:00:00.3"),
            generate_sample_audio(1, 48_000, dtype=np.float64)[0][
                4_800 * 3 : 4_800 * 4
            ],
            id="subpart",
        ),
    ],
    indirect=["audio_files"],
)
def test_split_data_frames(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    start_frame: int,
    stop_frame: int,
    expected_begin: pd.Timestamp,
    expected_data: np.ndarray,
) -> None:
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
    )
    ad = dataset.data[0].split_frames(start_frame, stop_frame)

    assert ad.begin == expected_begin
    assert np.array_equal(ad.get_value(), expected_data)
