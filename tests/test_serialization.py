from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming, hann

import osekit.core
import osekit.core.audio_file
from osekit.audio_backend.audio_file_manager import AudioFileManager
from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    TIMESTAMP_FORMATS_EXPORTED_FILES,
)
from osekit.core.audio_data import AudioData
from osekit.core.audio_dataset import AudioDataset
from osekit.core.audio_file import AudioFile
from osekit.core.frequency_scale import Scale, ScalePart
from osekit.core.instrument import Instrument
from osekit.core.json_serializer import (
    _anchor,
    _deserialize_path,
    _serialize_path,
    deserialize_paths,
    path_anchor,
    serialize_paths,
)
from osekit.core.spectro_data import SpectroData
from osekit.core.spectro_dataset import SpectroDataset
from osekit.core.spectro_file import SpectroFile
from osekit.utils.audio import Normalization
from tests.helpers.audio import MockedAudioFile


def test_audio_file_from_dict_depends_on_available_info(
    audio_files: tuple[list[AudioFile], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_files, _ = audio_files
    af = audio_files[0]
    minimum = {
        "begin": af.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED),
        "end": af.end.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED),
        "path": af.path,
    }
    full = minimum | {
        "sample_rate": af.sample_rate,
        "channels": af.channels,
    }

    afm_calls = [0]

    info_original = AudioFileManager.info
    afm = osekit.core.audio_file.afm

    def patch_afm_info(*args, **kwargs) -> tuple[int, int, int]:
        afm_calls[0] += 1
        return info_original(self=afm, *args, **kwargs)

    monkeypatch.setattr(afm, "info", patch_afm_info)

    # AudioFile deserialization with minimum info should call afm info to read metadata
    AudioFile.from_dict(minimum)
    assert afm_calls[0] == 1

    # AudioFile deserialization with full info should not call afm info
    AudioFile.from_dict(full)
    assert afm_calls[0] == 1


def test_audio_file_to_dict_should_contain_afm_info(
    audio_files: tuple[list[AudioFile], Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_files, _ = audio_files
    af = audio_files[0]

    minimum = {
        "begin": af.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED),
        "end": af.end.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED),
        "path": af.path,
    }
    full = minimum | {
        "sample_rate": af.sample_rate,
        "channels": af.channels,
    }

    assert all(key in af.to_dict() for key in full)


@pytest.mark.parametrize(
    ("audio_files", "begin", "end", "sample_rate", "normalization"),
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
            Normalization.RAW,
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
            48_000,
            Normalization.ZSCORE,
            id="normalized_audio",
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
            Normalization.RAW,
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
            Normalization.RAW,
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
            Normalization.RAW,
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
            Normalization.RAW,
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
            Normalization.RAW,
            id="two_files_with_gap",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00+0200"),
            },
            Timestamp("2024-01-01 12:00:01+0200"),
            Timestamp("2024-01-01 12:00:04+0200"),
            48_000,
            Normalization.RAW,
            id="localized_files",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00+0200"),
            },
            Timestamp("2024-01-01 12:00:01+0200"),
            Timestamp("2024-01-01 12:00:04+0200"),
            48_000,
            Normalization.DC_REJECT,
            id="localized_normalized_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_data_serialization(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    begin: Timestamp | None,
    end: Timestamp | None,
    sample_rate: float,
    normalization: Normalization,
) -> None:
    audio_files, _ = audio_files

    ad = AudioData.from_files(
        audio_files,
        begin=begin,
        end=end,
        sample_rate=sample_rate,
        normalization=normalization,
    )

    assert np.array_equal(ad.get_value(), AudioData.from_dict(ad.to_dict()).get_value())


def test_audio_data_channels_serialization(monkeypatch: pytest.MonkeyPatch) -> None:
    mocked_value = np.array(
        [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ],
    )

    af = MockedAudioFile(
        mocked_value=mocked_value,
    )

    def mocked_make_file(**kwargs: dict) -> MockedAudioFile:
        return MockedAudioFile(mocked_value=mocked_value, **kwargs)

    monkeypatch.setattr(AudioData, "_make_file", mocked_make_file)

    ad = AudioData.from_files([af], channels=[0, 2])

    assert np.array_equal(AudioData.from_dict(ad.to_dict()).channels, [0, 2])


@pytest.mark.parametrize(
    ("audio_files", "data_duration", "sample_rate", "normalization", "name"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
            },
            Timedelta(seconds=1),
            48_000,
            Normalization.RAW,
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
            Normalization.RAW,
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
            Normalization.RAW,
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
            Normalization.RAW,
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
            Normalization.RAW,
            "merriweather post pavilion",
            id="named_ads",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00+0200"),
            },
            Timedelta(seconds=1),
            48_000,
            Normalization.RAW,
            "merriweather post pavilion",
            id="localized_ads",
        ),
    ],
    indirect=["audio_files"],
)
def test_audio_dataset_serialization(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    data_duration: Timestamp | None,
    sample_rate: float | list[float],
    normalization: Normalization,
    name: str | None,
) -> None:
    audio_files, _ = audio_files
    begin = min(af.begin for af in audio_files)

    strptime_format = (
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
        if begin.tzinfo is None
        else TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED
    )

    ads = AudioDataset.from_folder(
        tmp_path,
        strptime_format=strptime_format,
        data_duration=data_duration,
        normalization=normalization,
        name=name,
    )

    # NO TIMESTAMP IN THE NAME SHOULD STILL BE DESERIALIZED PROPERLY
    for idx, ad in list(enumerate(ads.data))[::2]:
        ad.name = str(idx)

    assert ads.begin == begin

    if type(sample_rate) is list:
        for data, sr in zip(ads.data, sample_rate, strict=False):
            data.sample_rate = sr
    else:
        ads.sample_rate = sample_rate

    assert all(
        np.array_equal(ad.get_value(), ad2.get_value())
        for ad, ad2 in zip(
            ads.data,
            AudioDataset.from_dict(ads.to_dict()).data,
            strict=False,
        )
    )

    ads.write_json(tmp_path)

    ads2 = AudioDataset.from_json(tmp_path / f"{ads}.json")

    assert str(ads) == str(ads2)
    assert ads.name == ads2.name
    assert ads.sample_rate == ads2.sample_rate
    assert ads.begin == ads2.begin
    assert ads.normalization == ads2.normalization

    zipped = zip(ads.data, ads2.data, strict=True)

    assert all(ad1.name == ad2.name for ad1, ad2 in zipped)

    assert all(np.array_equal(ad.get_value(), ad2.get_value()) for ad, ad2 in zipped)


@pytest.mark.parametrize(
    (
        "audio_files",
        "begin",
        "end",
        "sft",
        "sample_rate",
        "instrument",
        "v_lim",
        "colormap",
    ),
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
                win=hamming(1024),
                hop=1024,
                fs=48_000,
                mfft=1024,
                scale_to="magnitude",
            ),
            48_000,
            None,
            None,
            None,
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
                win=hamming(1024),
                hop=1024,
                fs=48_000,
                mfft=1024,
                scale_to="magnitude",
            ),
            48_000,
            Instrument(end_to_end_db=150.0),
            None,
            None,
            id="specified_instrument",
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
                win=hamming(1024),
                hop=1024,
                fs=48_000,
                mfft=1024,
                scale_to="magnitude",
            ),
            48_000,
            None,
            (-50.0, 0.0),
            None,
            id="specified_v_lim",
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
                win=hamming(1024),
                hop=1024,
                fs=48_000,
                mfft=1024,
                scale_to="magnitude",
            ),
            48_000,
            Instrument(end_to_end_db=150.0),
            (0.0, 150.0),
            None,
            id="specified_v_lim_and_instrument",
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
                win=hamming(1024),
                hop=1024,
                fs=48_000,
                mfft=1024,
                scale_to="psd",
            ),
            48_000,
            None,
            None,
            None,
            id="psd_spectrum",
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
            ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024),
            48_000,
            None,
            None,
            "inferno",
            id="different_colormap",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00+0200"),
            },
            None,
            None,
            ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024),
            48_000,
            None,
            None,
            None,
            id="timezone_aware",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectro_data_serialization(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    begin: Timestamp | None,
    end: Timestamp | None,
    sft: ShortTimeFFT,
    sample_rate: float,
    instrument: Instrument | None,
    v_lim: tuple[float, float] | None,
    colormap: str | None,
) -> None:
    af, _ = audio_files

    ad = AudioData.from_files(
        af,
        begin=begin,
        end=end,
        sample_rate=sample_rate,
        instrument=instrument,
    )

    # SpectroData linked to AudioData

    sd = SpectroData.from_audio_data(ad, sft, colormap=colormap)
    sd2 = SpectroData.from_dict(sd.to_dict(embed_sft=True))

    assert np.allclose(
        sd.get_value(),
        sd2.get_value(),
    )

    assert np.allclose(
        sd._to_db(sd.get_value()),
        sd2._to_db(sd2.get_value()),
    )

    assert sd.db_ref == sd2.db_ref
    assert sd.v_lim == sd2.v_lim

    assert sd.colormap == sd2.colormap

    # SpectroData linked to SpectroFiles

    sd.write(tmp_path, link=True)

    sfs = [
        SpectroFile(file, strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES)
        for file in tmp_path.glob("*.npz")
    ]

    sd2 = SpectroData.from_files(sfs)
    sd2 = SpectroData.from_dict(sd2.to_dict(embed_sft=True))

    assert np.array_equal(
        sd.get_value(),
        sd2.get_value(),
    )

    assert np.array_equal(
        sd._to_db(sd.get_value()),
        sd2._to_db(sd2.get_value()),
    )

    assert sd.db_ref == sd2.db_ref
    assert sd.v_lim == sd2.v_lim

    # Linked file from dict

    assert SpectroData.from_dict(sd.to_dict(embed_sft=True)).files == sd.files


def test_spectro_data_serialization_channel() -> None:
    ad = AudioData.from_files(
        [
            MockedAudioFile(
                mocked_value=np.empty(shape=(100, 3)),
                sample_rate=100,
            ),
        ],
        channels=[0, 2],
    )

    sd = SpectroData.from_audio_data(
        ad,
        fft=ShortTimeFFT(win=hamming(48), hop=48, fs=100),
    )
    sd.audio_channel = 2

    assert SpectroData.from_dict(sd.to_dict()).audio_channel == sd.audio_channel


@pytest.mark.parametrize(
    (
        "audio_files",
        "data_duration",
        "sample_rate",
        "sfts",
        "instrument",
        "v_lim",
        "colormap",
        "frequency_scale",
        "name",
    ),
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
            None,
            None,
            None,
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
            None,
            None,
            None,
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
            [ShortTimeFFT(win=hamming(1024), hop=1024, fs=48_000, mfft=1024)],
            Instrument(end_to_end_db=150.0),
            None,
            None,
            None,
            None,
            id="specified_instrument",
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
            (-100.0, 0.0),
            None,
            None,
            None,
            id="specified_v_lim",
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
            Instrument(end_to_end_db=150.0),
            (0.0, 150.0),
            None,
            None,
            None,
            id="specified_instrument_and_v_lim",
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
            None,
            None,
            None,
            None,
            id="ten_spectro_data_two_sfts",
        ),
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
            None,
            "inferno",
            None,
            None,
            id="specified_colormap",
        ),
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
            None,
            None,
            Scale(
                [
                    ScalePart(0.0, 0.5, 100, 48_000, "log"),
                    ScalePart(0.0, 0.5, 0, 24_000, "lin"),
                ],
            ),
            None,
            id="specified_scale",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectro_dataset_serialization(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    data_duration: Timestamp | None,
    sample_rate: float | list[float],
    sfts: list[ShortTimeFFT],
    instrument: Instrument | None,
    v_lim: tuple[float, float] | None,
    colormap: str | None,
    frequency_scale: Scale | None,
    name: str | None,
) -> None:
    audio_files, _ = audio_files
    begin = min(af.begin for af in audio_files)

    strptime_format = (
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
        if begin.tzinfo is None
        else TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED
    )

    ads = AudioDataset.from_folder(
        tmp_path,
        strptime_format=strptime_format,
        data_duration=data_duration,
        instrument=instrument,
    )

    ads.sample_rate = sample_rate

    sds = SpectroDataset.from_audio_dataset(
        audio_dataset=ads,
        fft=sfts[0],
        name=name,
        colormap=colormap,
        v_lim=v_lim,
        scale=frequency_scale,
    )
    for idx, sd in enumerate(sds.data):  # Apply different SFTs to the sds data
        sd.fft = sfts[idx // (len(sds.data) // len(sfts))]

    sds.write_json(tmp_path)

    sds2 = SpectroDataset.from_json(tmp_path / f"{sds}.json")

    assert str(sds) == str(sds2)
    assert sds.name == sds2.name
    assert sds.colormap == sds2.colormap
    assert sds.scale == sds2.scale
    assert sds.begin == sds2.begin
    assert all(
        np.array_equal(sd.get_value(), sd2.get_value())
        for sd, sd2 in zip(sds.data, sds2.data, strict=False)
    )
    assert all(
        sd.db_ref == sd2.db_ref for sd, sd2 in zip(sds.data, sds2.data, strict=False)
    )
    assert all(
        np.array_equal(sd._to_db(sd.get_value()), sd2._to_db(sd2.get_value()))
        for sd, sd2 in zip(sds.data, sds2.data, strict=False)
    )

    # Deserialized spectro data that share a same SFT should point to the same instance
    if len(sds2.data) > 1:
        assert sds2.data[0].fft == sds2.data[1].fft

    # SpectroDataset from npz files

    for sd in sds.data:
        sd.write(tmp_path)

    sds3 = SpectroDataset.from_folder(
        tmp_path,
        strptime_format=strptime_format,
        data_duration=data_duration,
    )

    assert all(
        np.array_equal(sd.get_value(), sd3.get_value())
        for sd, sd3 in zip(sds.data, sds3.data, strict=False)
    )
    assert all(
        sd.db_ref == sd3.db_ref for sd, sd3 in zip(sds.data, sds3.data, strict=False)
    )
    assert all(
        np.array_equal(sd._to_db(sd.get_value()), sd3._to_db(sd3.get_value()))
        for sd, sd3 in zip(sds.data, sds3.data, strict=False)
    )
    assert sds.begin == sds3.begin

    # Linked SpectroDataset JSON serialization

    sds.write(tmp_path / "linked", link=True)

    assert not any(sd.is_empty for sd in sds.data)

    sds.write_json(tmp_path / "linked")
    sds4 = SpectroDataset.from_json(tmp_path / "linked" / f"{sds}.json")

    assert all(
        sd1.files == sd2.files for sd1, sd2 in zip(sds.data, sds4.data, strict=True)
    )
    assert str(sds) == str(sds4)
    assert sds.name == sds4.name
    assert sds.colormap == sds4.colormap
    assert sds.scale == sds4.scale
    assert sds.begin == sds4.begin


"""
    >>> str(PureWindowsPath(relative_to_absolute(target_path=r'relative\target', root_path=r'C:\absolute\root')))
    'C:\\absolute\\root\\relative\\target'
    >>> str(PureWindowsPath(relative_to_absolute(target_path=r'D:\absolute\\path\root\target', root_path=r'C:\absolute\root')))
    'C:\\absolute\\path\\root\\target'
    >>> str(PurePosixPath(relative_to_absolute(target_path=r'C:/user/cool/data/audio/fun.wav', root_path=r'/home/project/cool/processed/stuff')))
    '/home/project/cool/data/audio/fun.wav'
"""


def can_symlink(tmp_path: Path) -> bool:
    try:
        test_link = tmp_path / "link_test"
        test_link.symlink_to(tmp_path)
        test_link.unlink()
    except OSError:
        return False
    return True


def test_path_anchor() -> None:
    assert _anchor.get() is None  # Default

    @path_anchor(anchor=Path(r"cool/anchor"))
    def test_anchor() -> None:
        assert _anchor.get() == Path(r"cool/anchor")

    test_anchor()

    assert _anchor.get() is None


@pytest.mark.parametrize(
    ("value", "root", "expected"),
    [
        pytest.param(
            "cool/fun/top.wav",
            "cool",
            "fun/top.wav",
            id="value_is_relative_to_root",
        ),
        pytest.param(
            "cool/fun/top.wav",
            "cool/jazz",
            "../fun/top.wav",
            id="value_requires_walk_up",
        ),
    ],
)
def test_serialize_without_anchor_relative(
    tmp_path: Path,
    value: str,
    root: str,
    expected: str,
) -> None:
    value, root = (tmp_path / relpath for relpath in (value, root))

    assert _serialize_path(value=value, root_path=root) == expected


def test_serialize_without_anchor_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def patch_relpath(*args, **kwargs) -> None:
        """Patch relpatch to raise a ``ValueError``.

        This simulates an error when e.g. trying to express a path relative
        to a ``root_path`` in another partition."""
        raise ValueError

    monkeypatch.setattr("osekit.core.json_serializer.os.path.relpath", patch_relpath)

    value = tmp_path / "fun" / "top.wav"

    # Not important as it will raise in patch_relpath whatever root_path is
    root_path = Path(r"cool")

    # Absolute path since value can't be expressed as relative to root_path
    expected = value.as_posix()

    assert _serialize_path(value=value, root_path=root_path) == expected


def test_serialize_path_within_anchor(tmp_path: Path) -> None:
    value = tmp_path / "fun" / "top.wav"
    anchor = tmp_path
    root_path = tmp_path / "fun"
    expected = "top.wav"

    with path_anchor(anchor=anchor):
        assert _serialize_path(value=value, root_path=root_path) == expected


def test_serialize_path_outside_anchor(tmp_path: Path) -> None:
    value = tmp_path / "fun" / "top.wav"
    anchor = tmp_path / "cool"
    root_path = tmp_path / "cool" / "fun"
    expected = value.as_posix()  # Absolute because we're outside the anchor

    with path_anchor(anchor=anchor):
        assert _serialize_path(value=value, root_path=root_path) == expected


def test_deserialize_absolute_path_bypasses_root(tmp_path: Path) -> None:
    value = tmp_path / "baston" / "saphir.wav"
    root = tmp_path / "fun" / "cool"

    result = _deserialize_path(value=value.as_posix(), root_path=root)

    assert result == value


def test_deserialize_relative_path(tmp_path: Path) -> None:
    value = "baston/saphir.wav"
    root = tmp_path / "fun"

    result = _deserialize_path(value=value, root_path=root)

    assert result == tmp_path / "fun" / "baston" / "saphir.wav"


def test_deserialize_relative_path_walk_up(tmp_path: Path) -> None:
    value = "../../baston/saphir.wav"
    root = tmp_path / "wilde" / "jagd"

    result = _deserialize_path(value=value, root_path=root)

    assert result == tmp_path / "baston/saphir.wav"


@pytest.mark.parametrize("use_anchor", [True, False])
def test_serialize_deserialize_round_trip(tmp_path: Path, use_anchor: bool) -> None:
    project = tmp_path / "project"
    remote = tmp_path / "remote"  # Emulates another partition
    root = project / "processed" / "jsons"

    inside = project / "data" / "audio" / "a1.wav"
    outside = remote / "audio" / "b.wav"

    context = path_anchor(anchor=project) if use_anchor else contextlib.nullcontext()

    with context:
        serialized_in = _serialize_path(value=inside, root_path=root)
        serialized_out = _serialize_path(value=outside, root_path=root)

    assert serialized_in == "../../data/audio/a1.wav"
    assert serialized_out == (
        outside.as_posix() if use_anchor else "../../../remote/audio/b.wav"
    )

    assert _deserialize_path(value=serialized_in, root_path=root) == inside.resolve()
    assert _deserialize_path(value=serialized_out, root_path=root) == outside.resolve()


@pytest.mark.parametrize("use_anchor", [True, False])
def test_serialize_dictionary_paths(tmp_path: Path, use_anchor: bool) -> None:
    project_path = tmp_path / "project"
    root_path = project_path / "fun"

    deserialized = {
        "path": tmp_path / "outside_project" / "audio.wav",
        "folder": project_path / "cool" / "audio.wav",
        "json": root_path / "audio.wav",
        "not_a_path": 16,
        "nested_dictionary": {
            "path": root_path / "audio.wav",
        },
    }

    serialized = {
        "path": (tmp_path / "outside_project" / "audio.wav").as_posix(),
        "folder": "../cool/audio.wav",  # Relative because in anchor
        "json": "audio.wav",
        "not_a_path": 16,
        "nested_dictionary": {
            "path": "audio.wav",
        },
    }

    input_copy = dict(deserialized)
    context = (
        path_anchor(anchor=project_path) if use_anchor else contextlib.nullcontext()
    )

    # Path outside project: absolute if we use project as anchor
    # relative if we don't use anchors.
    serialized["path"] = (
        (tmp_path / "outside_project" / "audio.wav").as_posix()
        if use_anchor
        else "../../outside_project/audio.wav"
    )

    with context:
        serialize_paths(
            data=input_copy,
            root_path=root_path,
        )
        assert input_copy == serialized


def test_dictionary_paths_deserialization(tmp_path: Path) -> None:
    input_dict = {
        "path": r"C:/windows_like/absolute_path",
        "folder": r"/posix/absolute_path",
        "json": r"relative/path",
        "not_a_path": 16,
        "nested_dictionary": {
            "path": r"relative/path",
        },
    }

    deserialized_dict = {
        "path": Path(r"C:/windows_like/absolute_path"),
        "folder": Path(r"/posix/absolute_path"),
        "json": tmp_path / "root_folder/relative/path",
        "not_a_path": 16,
        "nested_dictionary": {
            "path": tmp_path / "root_folder/relative/path",
        },
    }

    deserialize_paths(
        data=input_dict,
        root_path=tmp_path / "root_folder",
    )

    assert input_dict == deserialized_dict
