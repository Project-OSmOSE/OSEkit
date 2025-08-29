from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from pandas import Timestamp

import osekit
from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    resample_quality_settings,
)
from osekit.core_api import audio_file_manager as afm
from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_dataset import AudioDataset
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.audio_item import AudioItem
from osekit.utils import audio_utils
from osekit.utils.audio_utils import generate_sample_audio, normalizations


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
        pytest.param(
            {
                "duration": 0.05,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            id="localized_audio_file",
        ),
    ],
    indirect=True,
)
def test_audio_file_timestamps(
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
) -> None:
    audio_files, request = audio_files
    duration = request.param["duration"]
    date_begin = request.param["date_begin"]

    for audio_file in audio_files:
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
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    start: pd.Timestamp,
    stop: pd.Timestamp,
    expected: np.ndarray,
) -> None:
    files, request = audio_files
    assert np.allclose(files[0].read(start, stop), expected, atol=1e-7)


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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=500_000,
                tz="+0200",
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=0,
                microsecond=600_000,
                tz="+0200",
            ),
            generate_sample_audio(nb_files=1, nb_samples=48_000)[0][24_000:28_800],
            id="localized_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_item(
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    start: pd.Timestamp | None,
    stop: pd.Timestamp | None,
    expected: np.ndarray,
) -> None:
    files, request = audio_files
    item = AudioItem(files[0], start, stop)
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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
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
                tz="+0200",
            ),
            pd.Timestamp(
                year=2024,
                month=1,
                day=1,
                hour=12,
                minute=0,
                second=1,
                microsecond=200_000,
                tz="+0200",
            ),
            generate_sample_audio(nb_files=1, nb_samples=48_000 * 2)[0][38_400:57_600],
            id="localized_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_data(
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    start: pd.Timestamp | None,
    stop: pd.Timestamp | None,
    expected: np.ndarray,
) -> None:
    files, request = audio_files
    data = AudioData.from_files(files, begin=start, end=stop)
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
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
) -> None:
    audio_files, _ = audio_files
    ad = AudioData.from_files(audio_files)
    assert np.array_equal(sf.read(audio_files[0].path)[0], ad.get_value())


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
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    start: pd.Timestamp | None,
    stop: pd.Timestamp | None,
    sample_rate: int,
    expected_nb_samples: int,
) -> None:
    audio_files, request = audio_files
    data = AudioData.from_files(audio_files, begin=start, end=stop)
    data.sample_rate = sample_rate
    assert data.get_value().shape[0] == expected_nb_samples


@pytest.mark.parametrize(
    ("audio_files", "downsampling_quality", "upsampling_quality"),
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
            id="default_qualities",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            "HQ",
            None,
            id="downsample_quality_to_HQ",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            "QQ",
            id="upsample_quality_to_QQ",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            "VHQ",
            "VHQ",
            id="both_qualities_to_VHQ",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_resample_quality(
    audio_files: pytest.fixture,
    monkeypatch: pytest.MonkeyPatch,
    downsampling_quality: str | None,
    upsampling_quality: str | None,
) -> None:
    importlib.reload(osekit.config)

    files, _ = audio_files
    af = files[0]

    downsampling_default = resample_quality_settings["downsample"]
    upsampling_default = resample_quality_settings["upsample"]

    if downsampling_quality is not None:
        osekit.config.resample_quality_settings["downsample"] = downsampling_quality
    if upsampling_quality is not None:
        osekit.config.resample_quality_settings["upsample"] = upsampling_quality

    def resample_mkptch(
        data: np.ndarray,
        origin_sr: float,
        target_sr: float,
    ) -> np.ndarray:
        return (
            osekit.config.resample_quality_settings["upsample"]
            if target_sr > origin_sr
            else osekit.config.resample_quality_settings["downsample"]
        )

    monkeypatch.setattr(audio_utils, "resample", resample_mkptch)

    ad = AudioData.from_files([af])

    downsampling_frequency, upsampling_frequency = (
        ratio * ad.sample_rate for ratio in (0.5, 1.5)
    )

    assert audio_utils.resample(
        ad.get_value(),
        ad.sample_rate,
        downsampling_frequency,
    ) == (
        downsampling_quality
        if downsampling_quality is not None
        else downsampling_default
    )
    assert audio_utils.resample(
        ad.get_value(),
        ad.sample_rate,
        upsampling_frequency,
    ) == (upsampling_quality if upsampling_quality is not None else upsampling_default)


@pytest.mark.parametrize(
    ("audio_files", "normalization"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            "raw",
            id="no_normalization",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            "dc_reject",
            id="dc_reject",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 10,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            "zscore",
            id="z_score",
        ),
    ],
    indirect=["audio_files"],
)
def test_normalize_audio_data(
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    normalization: Literal["raw", "dc_reject", "zscore"],
) -> None:
    afs, _ = audio_files

    raw_data = np.linspace(0.0, 1.0, 10)
    normalized_data = normalizations[normalization](raw_data)

    if normalization == "raw":
        assert np.array_equal(raw_data, normalized_data)
    else:
        assert not np.array_equal(raw_data, normalized_data)

    # AudioData
    ad = AudioData.from_files(afs, normalization=normalization)
    assert np.array_equal(ad.get_value(), normalized_data)

    # AudioDataset
    ads = AudioDataset.from_files(afs, normalization=normalization)
    assert ads.data[0].normalization == normalization
    assert np.array_equal(ads.data[0].get_value(), normalized_data)


@pytest.mark.parametrize(
    ("audio_files", "begin", "end", "mode", "duration", "expected_audio_data"),
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
            "timedelta_total",
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
            "timedelta_total",
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
            "timedelta_total",
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
            "timedelta_total",
            pd.Timedelta(seconds=1),
            generate_sample_audio(nb_files=2, nb_samples=48_000),
            id="overlapping_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "inter_file_duration": 0,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
                "series_type": "increase",
            },
            None,
            None,
            "files",
            None,
            generate_sample_audio(
                nb_files=3,
                nb_samples=48_000,
                series_type="increase",
            ),
            id="files_mode_without_overlap",
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
            "files",
            None,
            [
                generate_sample_audio(nb_files=1, nb_samples=96_000)[0][0:48_000],
                generate_sample_audio(nb_files=1, nb_samples=96_000)[0][48_000:],
            ],
            id="files_mode_with_gap",
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
            "files",
            None,
            [
                generate_sample_audio(nb_files=1, nb_samples=48_000)[0][0:24_000],
                generate_sample_audio(nb_files=1, nb_samples=48_000)[0][0:24_000],
                generate_sample_audio(nb_files=1, nb_samples=48_000)[0],
            ],
            id="files_mode_with_overlap",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_dataset_from_folder(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    begin: pd.Timestamp | None,
    end: pd.Timestamp | None,
    mode: Literal["files", "timedelta_total", "timedelta_file"],
    duration: pd.Timedelta | None,
    expected_audio_data: list[np.ndarray],
) -> None:
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        begin=begin,
        end=end,
        mode=mode,
        data_duration=duration,
    )
    for idx, data in enumerate(dataset.data):
        vs = data.get_value()
        assert np.array_equal(expected_audio_data[idx], vs)


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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 3,
                "inter_file_duration": -0.5,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
                "series_type": "repeat",
            },
            pd.Timestamp("2024-01-01 12:00:00+0200"),
            pd.Timestamp("2024-01-01 12:00:02+0200"),
            pd.Timedelta(seconds=1),
            generate_sample_audio(nb_files=2, nb_samples=48_000),
            id="localized_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_dataset_from_files(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    begin: pd.Timestamp | None,
    end: pd.Timestamp | None,
    duration: pd.Timedelta | None,
    expected_audio_data: list[np.ndarray],
) -> None:
    strptime_format = (
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
        if (begin is None or begin.tz is None)
        else TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED
    )
    afs = [
        AudioFile(f, strptime_format=strptime_format) for f in tmp_path.glob("*.wav")
    ]
    dataset = AudioDataset.from_files(
        afs,
        begin=begin,
        end=end,
        data_duration=duration,
    )
    assert all(
        np.array_equal(data.get_value(), expected)
        for (data, expected) in zip(dataset.data, expected_audio_data, strict=False)
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
                match="No valid file found in ",
            ),
            id="no_file",
        ),
        pytest.param(
            {"nb_files": 0},
            [],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".flac",
            ],
            [],
            pytest.raises(
                FileNotFoundError,
                match="No valid file found in ",
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
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
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
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
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
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".flac",
            ],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".csv",
            ],
            pytest.raises(
                FileNotFoundError,
                match="No valid file found in ",
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
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".wav",
                pd.Timestamp("2000-01-01 00:00:10").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                )
                + ".flac",
            ],
            [
                pd.Timestamp("2000-01-01 00:00:00").strftime(
                    format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
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
    caplog: pytest.LogCaptureFixture,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    expected_audio_data: list[np.ndarray],
    corrupted_audio_files: list[str],
    non_audio_files: list[str],
    error: type[Exception],
) -> None:
    for corrupted_file in corrupted_audio_files:
        (tmp_path / corrupted_file).open("a").close()  # Write empty audio files.

    with caplog.at_level(logging.WARNING):
        if error is not None:
            with error as e:
                assert (
                    AudioDataset.from_folder(
                        tmp_path,
                        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
                    )
                    == e
                )
        else:
            dataset = AudioDataset.from_folder(
                tmp_path,
                strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
            )
            assert all(
                np.array_equal(data.get_value(), expected)
                for (data, expected) in zip(
                    dataset.data,
                    expected_audio_data,
                    strict=False,
                )
            )
        assert all(f in caplog.text for f in corrupted_audio_files)


@pytest.mark.parametrize(
    ("audio_files", "subtype", "link", "expected_audio_data"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            "DOUBLE",
            False,
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
            False,
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
            False,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="padded_file",
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
            True,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="link_to_written_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            "DOUBLE",
            False,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="localized_written_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            "DOUBLE",
            True,
            generate_sample_audio(1, 48_000, dtype=np.float64),
            id="link_to_localized_written_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_write_files(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    subtype: str,
    link: bool,
    expected_audio_data: list[np.ndarray],
) -> None:
    begin = min(af.begin for af in audio_files[0])
    strptime_format = (
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
        if (begin is None or begin.tz is None)
        else TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED
    )
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=strptime_format,
    )
    output_path = tmp_path / "output"
    dataset.write(output_path, subtype=subtype, link=link)
    for data in dataset.data:
        assert f"{data}.wav" in [f.name for f in output_path.glob("*.wav")]
        assert np.allclose(data.get_value(), sf.read(output_path / f"{data}.wav")[0])

        if link:
            assert str(next(iter(data.files)).path) == str(output_path / f"{data}.wav")


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
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )
    for data in dataset.data:
        subdata_shape = data.shape // nb_subdata
        for subdata, data_range in zip(
            data.split(nb_subdata),
            range(0, data.shape, subdata_shape),
            strict=False,
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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 144_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            30,
            60,
            pd.Timestamp("2024-01-01 12:00:00.000208334"),
            generate_sample_audio(1, 144_000, dtype=np.float64)[0][30:60],
            id="higher_fs",
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
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )
    ad = dataset.data[0].split_frames(start_frame, stop_frame)

    assert ad.begin == expected_begin
    assert np.array_equal(ad.get_value(), expected_data)


@pytest.mark.parametrize(
    "audio_files",
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            id="move_one_audio_file",
        ),
    ],
    indirect=["audio_files"],
)
def test_move_audio_file(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
) -> None:
    ad = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    ).data[0]
    af = next(iter(ad.files))

    destination_folder = tmp_path / "cool"

    old_path = str(af.path)
    af_name = af.path.name

    # Moving file without opening it first
    af.move(destination_folder)

    assert (destination_folder / af_name).exists()
    assert not Path(old_path).exists()
    assert afm.opened_file is None

    # Accessing the file at the new path
    ad.get_value()

    assert afm.opened_file is not None
    assert afm.opened_file.name == str(af.path)

    # Moving it back after opening it in the afm
    # afm should close the file to allow the moving
    af.move(tmp_path)

    assert afm.opened_file is None
    assert not (destination_folder / af_name).exists()
    assert Path(old_path).exists()

    # Reading the file again
    ad.get_value()

    assert afm.opened_file is not None
    assert afm.opened_file.name == str(af.path)


@pytest.mark.parametrize(
    (
        "audio_files",
        "ad1_begin",
        "ad2_begin",
        "ad1_end",
        "ad2_end",
        "ad1_sample_rate",
        "ad2_sample_rate",
        "expected",
    ),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:01"),
            48_000,
            48_000,
            True,
            id="equal_data",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:02"),
            Timestamp("2024-01-01 12:00:02"),
            48_000,
            48_000,
            False,
            id="different_begin",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:02"),
            48_000,
            48_000,
            False,
            id="different_end",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:02"),
            Timestamp("2024-01-01 12:00:02"),
            48_000,
            24_000,
            False,
            id="different_sample_rate",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": -1.0,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:00"),
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:01"),
            48_000,
            48_000,
            False,
            id="different_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_data_equality(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    ad1_begin: Timestamp,
    ad2_begin: Timestamp,
    ad1_end: Timestamp,
    ad2_end: Timestamp,
    ad1_sample_rate: float,
    ad2_sample_rate: float,
    expected: bool,
) -> None:
    afs = iter(
        AudioFile(f, strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED)
        for f in tmp_path.glob("*.wav")
    )
    af1 = next(afs)
    af2 = next(afs, af1)

    ad1 = AudioData.from_files(
        [af1],
        begin=ad1_begin,
        end=ad1_end,
    )
    ad1.sample_rate = ad1_sample_rate

    ad2 = AudioData.from_files(
        [af2],
        begin=ad2_begin,
        end=ad2_end,
    )
    ad2.sample_rate = ad2_sample_rate

    assert (ad1 == ad2) == expected

    # AudioDataset scope
    ads1 = AudioDataset([ad1])
    ads2 = AudioDataset([ad2])
    assert (ads1 == ads2) == expected
    ads2.sample_rate = 500

    # AudioDataset equality should account for sample rate
    assert ads1 != ads2
