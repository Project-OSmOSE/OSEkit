from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming, hann

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES, TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.core_api.audio_data import AudioData
from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.audio_file import AudioFile
from OSmOSE.core_api.spectro_data import SpectroData
from OSmOSE.core_api.spectro_dataset import SpectroDataset
from OSmOSE.core_api.spectro_file import SpectroFile

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


@pytest.mark.parametrize(
    ("audio_files", "data_duration", "sample_rate", "name"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            Timedelta(seconds=1),
            48_000,
            None,
            id="one_audio_data_one_file_no_resample",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
            },
            Timedelta(seconds=2),
            48_000,
            None,
            id="one_audio_data_two_files_no_resample",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
            },
            Timedelta(seconds=2),
            24_000,
            None,
            id="one_audio_data_two_files_downsample",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 4,
            },
            Timedelta(seconds=1),
            [12_000, 24_000, 48_000, 96_000],
            None,
            id="multiple_audio_data_different_sample_rates",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            Timedelta(seconds=1),
            48_000,
            "merriweather post pavilion",
            id="named_ads",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_dataset_serialization(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    data_duration: Timestamp | None,
    sample_rate: float | list[float],
    name: str | None,
) -> None:

    ads = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
        data_duration=data_duration,
        name=name,
    )

    if type(sample_rate) is list:
        for data, sr in zip(ads.data, sample_rate):
            data.sample_rate = sr
    else:
        ads.sample_rate = sample_rate

    assert all(
        np.array_equal(ad.get_value(), ad2.get_value())
        for ad, ad2 in zip(ads.data, AudioDataset.from_dict(ads.to_dict()).data)
    )

    ads.write_json(tmp_path)

    ads2 = AudioDataset.from_json(tmp_path / f"{ads}.json")

    assert str(ads) == str(ads2)
    assert ads.name == ads2.name
    assert ads.has_default_name == ads2.has_default_name
    assert ads.sample_rate == ads2.sample_rate

    assert all(
        np.array_equal(ad.get_value(), ad2.get_value())
        for ad, ad2 in zip(ads.data, ads2.data)
    )


@pytest.mark.parametrize(
    ("audio_files", "begin", "end", "sft", "sample_rate"),
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
            ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024),
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
            ShortTimeFFT(win=hamming(1024, sym=False), hop=1024, fs=24_000, mfft=1024),
            24_000,
            id="non_symetric_hamming_window",
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
            ShortTimeFFT(win=hann(1024), hop=1024, fs=24_000, mfft=1024),
            24_000,
            id="hann_window",
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
            ShortTimeFFT(win=hamming(1024), hop=512, fs=24_000, mfft=1024),
            24_000,
            id="full_file_downsample_and_overlap",
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
            ShortTimeFFT(win=hamming(1024), hop=1024, fs=96_000, mfft=2048),
            96_000,
            id="full_file_upsample_and_mfft_padding",
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
            ShortTimeFFT(win=hamming(1024), hop=512, fs=48_000, mfft=2048),
            48_000,
            id="file_part_and_overlap_and_padding",
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
            ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000, mfft=1024),
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
            ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000, mfft=1024),
            48_000,
            id="two_files_with_gap",
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
            ShortTimeFFT(
                win=hamming(1024), hop=1024, fs=48_000, mfft=1024, scale_to="magnitude"
            ),
            48_000,
            id="magnitude_spectrum",
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
            ShortTimeFFT(
                win=hamming(1024), hop=1024, fs=48_000, mfft=1024, scale_to="psd"
            ),
            48_000,
            id="psd_spectrum",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectro_data_serialization(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    begin: Timestamp | None,
    end: Timestamp | None,
    sft: ShortTimeFFT,
    sample_rate: float,
) -> None:
    af = [
        AudioFile(f, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
        for f in tmp_path.glob("*.wav")
    ]

    ad = AudioData.from_files(af, begin=begin, end=end, sample_rate=sample_rate)

    # SpectroData linked to AudioData

    sd = SpectroData.from_audio_data(ad, sft)

    assert np.allclose(
        sd.get_value(),
        SpectroData.from_dict(sd.to_dict(embed_sft=True)).get_value(),
    )

    # SpectroData linked to SpectroFiles

    sd.write(tmp_path)

    sfs = [
        SpectroFile(file, strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES)
        for file in tmp_path.glob("*.npz")
    ]

    sd2 = SpectroData.from_files(sfs)

    assert np.array_equal(
        sd.get_value(),
        SpectroData.from_dict(sd2.to_dict(embed_sft=True)).get_value(),
    )


@pytest.mark.parametrize(
    ("audio_files", "data_duration", "sample_rate", "sfts", "name"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            Timedelta(seconds=1),
            48_000,
            [ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024)],
            None,
            id="one_spectro_data",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            Timedelta(seconds=0.1),
            48_000,
            [ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024)],
            None,
            id="ten_spectro_data_one_sft",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            Timedelta(seconds=0.1),
            48_000,
            [
                ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024),
                ShortTimeFFT(win=hann(1024), hop=512, fs=48_000, mfft=2048),
            ],
            None,
            id="ten_spectro_data_two_sfts",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectro_dataset_serialization(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    data_duration: Timestamp | None,
    sample_rate: float | list[float],
    sfts: list[ShortTimeFFT],
    name: str | None,
) -> None:

    ads = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
        data_duration=data_duration,
    )

    ads.sample_rate = sample_rate

    sds = SpectroDataset.from_audio_dataset(audio_dataset=ads, fft=sfts[0], name=name)
    for idx, sd in enumerate(sds.data):  # Apply different SFTs to the sds data
        sd.fft = sfts[idx // (len(sds.data) // len(sfts))]

    sds.write_json(tmp_path)

    sds2 = SpectroDataset.from_json(tmp_path / f"{sds}.json")

    assert str(sds) == str(sds2)
    assert sds.name == sds2.name
    assert sds.has_default_name == sds2.has_default_name
    assert all(
        np.array_equal(sd.get_value(), sd2.get_value())
        for sd, sd2 in zip(sds.data, sds2.data)
    )

    # Deserialized spectro data that share a same SFT should point to the same instance
    if len(sds2.data) > 1:
        assert sds2.data[0].fft == sds2.data[1].fft

    # SpectroDataset from npz files

    for sd in sds.data:
        sd.write(tmp_path)

    sds3 = SpectroDataset.from_files(
        [
            SpectroFile(file, strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES)
            for file in tmp_path.glob("*.npz")
        ],
        data_duration=data_duration,
    )

    assert all(
        np.array_equal(sd.get_value(), sd3.get_value())
        for sd, sd3 in zip(sds.data, sds3.data)
    )
