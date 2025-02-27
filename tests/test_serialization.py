from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pandas import Timestamp

from OSmOSE.config import TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.core_api.audio_data import AudioData
from OSmOSE.core_api.audio_file import AudioFile

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("audio_files", "begin", "end", "sample_rate"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            48_000,
            id="full_file_no_resample",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            24_000,
            id="full_file_downsample",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            96_000,
            id="full_file_upsample",
        ),
        pytest.param(
            {
                "duration": 3,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:02"),
            48_000,
            id="file_part",
        ),
        pytest.param(
            {
                "duration": 1.5,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:02"),
            24_000,
            id="two_files_with_resample",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:04"),
            48_000,
            id="two_files_with_gap",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_data_serialization(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    begin: Timestamp | None,
    end: Timestamp | None,
    sample_rate: float,
) -> None:
    af = [
        AudioFile(f, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
        for f in tmp_path.glob("*.wav")
    ]

    ad = AudioData.from_files(af, begin=begin, end=end, sample_rate=sample_rate)

    assert np.array_equal(ad.get_value(), AudioData.from_dict(ad.to_dict()).get_value())
