from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES, TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.data.audio_data import AudioData
from OSmOSE.data.audio_dataset import AudioDataset
from OSmOSE.data.audio_file import AudioFile
from OSmOSE.data.spectro_data import SpectroData
from OSmOSE.data.spectro_dataset import SpectroDataset
from OSmOSE.data.spectro_file import SpectroFile
from OSmOSE.utils.audio_utils import generate_sample_audio

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("audio_files", "original_audio_data", "sft"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            generate_sample_audio(1, 48_000, dtype=np.float64),
            ShortTimeFFT(hamming(1_024), 1024, 48_000),
            id="short_spectrogram",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            generate_sample_audio(1, 1_024, dtype=np.float64),
            ShortTimeFFT(hamming(1_024), 1024, 1_024),
            id="data_is_one_window_long",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectrogram_shape(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    original_audio_data: list[np.ndarray],
    sft: ShortTimeFFT,
) -> None:
    dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
    )
    spectro_dataset = SpectroDataset.from_audio_dataset(dataset, sft)
    for audio, spectro in zip(dataset.data, spectro_dataset.data):
        assert spectro.shape == spectro.get_value().shape
        assert spectro.shape == (sft.f.shape[0], sft.p_num(audio.shape))


@pytest.mark.parametrize(
    ("audio_files", "date_begin", "sft"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            pd.Timestamp("2024-01-01 12:00:00"),
            ShortTimeFFT(hamming(1_024), 1024, 48_000),
            id="second_precision",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
            },
            pd.Timestamp("2024-01-01 12:00:00.123"),
            ShortTimeFFT(hamming(512), 512, 1_024),
            id="millisecond_precision",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            pd.Timestamp("2024-01-01 12:00:00.123456"),
            ShortTimeFFT(hamming(1_024), 1_024, 48_000),
            id="microsecond_precision",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            pd.Timestamp("2024-01-01 12:00:00.123456789"),
            ShortTimeFFT(hamming(1_024), 1_024, 48_000),
            id="nanosecond_precision",
        ),
        pytest.param(
            {
                "duration": 1.123456789,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            pd.Timestamp("2024-01-01 12:00:00.123456789"),
            ShortTimeFFT(hamming(1_024), 1_024, 48_000),
            id="nanosecond_precision_end",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectro_parameters_in_npz_files(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    date_begin: pd.Timestamp,
    sft: ShortTimeFFT,
) -> None:

    af = next(AudioFile(f, begin=date_begin) for f in tmp_path.glob("*.wav"))

    ad = AudioData.from_files([af])
    sd = SpectroData.from_audio_data(ad, sft)
    sd.write(tmp_path / "npz")
    file = tmp_path / "npz" / f"{sd}.npz"
    sf = SpectroFile(file, strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES)

    assert sf.begin == ad.begin
    assert sf.end == ad.end
    assert np.array_equal(sf.freq, sft.f)
    assert sf.hop == sft.hop
    assert sf.mfft == sft.mfft
    assert sf.sample_rate == sft.fs
    nb_time_bins = sft.t(ad.shape).shape[0]
    assert np.array_equal(
        sf.time, np.arange(nb_time_bins) * ad.duration.total_seconds() / nb_time_bins
    )


@pytest.mark.parametrize(
    ("audio_files", "nb_chunks", "sft"),
    [
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            3,
            ShortTimeFFT(hamming(1_024), 1_024, 1_024),
            id="6_seconds_split_in_3",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            1,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="1_npz_file",
        ),
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            3,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="6_seconds_split_in_3_with_overlap",
        ),
        pytest.param(
            {
                "duration": 8,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            4,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="8_seconds_split_in_4",
        ),
        pytest.param(
            {
                "duration": 4,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            4,
            ShortTimeFFT(hamming(12_000), 12_000, 48_000),
            id="high_sr_no_overlap",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            3,
            ShortTimeFFT(hamming(12_000), 10_000, 48_000),
            id="high_sr_overlap",
        ),
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            6,
            ShortTimeFFT(hamming(1_024), 1_024, 1_024),
            id="6_seconds_split_in_6",
        ),
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            6,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="6_seconds_split_in_6_with_overlap",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectrogram_from_npz_files(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    nb_chunks: int,
    sft: ShortTimeFFT,
) -> None:
    afs = [
        AudioFile(f, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
        for f in tmp_path.glob("*.wav")
    ]

    ad = AudioData.from_files(afs)
    sd = SpectroData.from_audio_data(ad, sft)

    sd_split = sd.split(nb_chunks)

    for spectro in sd_split:
        spectro.write(tmp_path / "output")
    assert len(list((tmp_path / "output").glob("*.npz"))) == nb_chunks

    sds = SpectroDataset.from_folder(
        tmp_path / "output",
        TIMESTAMP_FORMAT_EXPORTED_FILES,
    )

    assert sds.begin == ad.begin
    assert sds.duration == ad.duration
    assert len(sds.data) == 1
    assert sds.data[0].shape == sds.data[0].get_value().shape

    assert sds.data[0].shape == (
        sft.f.shape[0],
        sft.p_num(int(ad.duration.total_seconds() * ad.sample_rate)),
    )

    assert np.allclose(sd.get_value(), sds.data[0].get_value())
