from __future__ import annotations

from pathlib import Path, PureWindowsPath

import numpy as np
import pytest
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming, hann

from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    TIMESTAMP_FORMATS_EXPORTED_FILES,
)
from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_dataset import AudioDataset
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.frequency_scale import Scale, ScalePart
from osekit.core_api.instrument import Instrument
from osekit.core_api.json_serializer import relative_to_absolute, set_path_reference
from osekit.core_api.spectro_data import SpectroData
from osekit.core_api.spectro_dataset import SpectroDataset
from osekit.core_api.spectro_file import SpectroFile
from osekit.utils.audio_utils import Normalization


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
    audio_files, request = audio_files
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
    assert ads.has_default_name == ads2.has_default_name
    assert ads.sample_rate == ads2.sample_rate
    assert ads.begin == ads2.begin
    assert ads.normalization == ads2.normalization

    assert all(
        np.array_equal(ad.get_value(), ad2.get_value())
        for ad, ad2 in zip(ads.data, ads2.data, strict=False)
    )


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
        sd.to_db(sd.get_value()),
        sd2.to_db(sd2.get_value()),
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
        sd.to_db(sd.get_value()),
        sd2.to_db(sd2.get_value()),
    )

    assert sd.db_ref == sd2.db_ref
    assert sd.v_lim == sd2.v_lim

    # Linked file from dict

    assert SpectroData.from_dict(sd.to_dict(embed_sft=True)).files == sd.files


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
    audio_files, request = audio_files
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
    assert sds.has_default_name == sds2.has_default_name
    assert sds.begin == sds2.begin
    assert all(
        np.array_equal(sd.get_value(), sd2.get_value())
        for sd, sd2 in zip(sds.data, sds2.data, strict=False)
    )
    assert all(
        sd.db_ref == sd2.db_ref for sd, sd2 in zip(sds.data, sds2.data, strict=False)
    )
    assert all(
        np.array_equal(sd.to_db(sd.get_value()), sd2.to_db(sd2.get_value()))
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
        np.array_equal(sd.to_db(sd.get_value()), sd3.to_db(sd3.get_value()))
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
    assert sds.has_default_name == sds4.has_default_name
    assert sds.begin == sds4.begin


"""
    >>> str(PureWindowsPath(relative_to_absolute(target_path=r'relative\target', root_path=r'C:\absolute\root')))
    'C:\\absolute\\root\\relative\\target'
    >>> str(PureWindowsPath(relative_to_absolute(target_path=r'D:\absolute\\path\root\target', root_path=r'C:\absolute\root')))
    'C:\\absolute\\path\\root\\target'
    >>> str(PurePosixPath(relative_to_absolute(target_path=r'C:/user/cool/data/audio/fun.wav', root_path=r'/home/dataset/cool/processed/stuff')))
    '/home/dataset/cool/data/audio/fun.wav'
"""


@pytest.mark.parametrize(
    ("target", "root", "expected"),
    [
        pytest.param(
            "saphir.wav",
            "C:/baston",
            "C:/baston/saphir.wav",
            id="relative_to_windows_root",
        ),
        pytest.param(
            "i/want/sky.wav",
            "C:/what/would",
            "C:/what/would/i/want/sky.wav",
            id="relative_tree_to_windows_root",
        ),
        pytest.param(
            "agency/group",
            "/the",
            "/the/agency/group",
            id="relative_tree_to_posix_root",
        ),
        pytest.param(
            "C:/user/dataset/file.csv",
            "C:/root/to/dataset",
            "C:/root/to/dataset/file.csv",
            id="absolute_windows_path_to_windows_root",
        ),
        pytest.param(
            "C:/user/dataset/file.csv",
            "C:/root/to/dataset/with/more",
            "C:/root/to/dataset/file.csv",
            id="absolute_windows_path_to_windows_root_takes_first_common_folder",
        ),
        pytest.param(
            "C:/user/datasets/audio/analysis/audio/audio/audio.wav",
            "C:/root/to/my_datasets/audio/some/stuff",
            "C:/root/to/my_datasets/audio/analysis/audio/audio/audio.wav",
            id="absolute_windows_path_to_windows_root_with_repeated_folders",
        ),
        pytest.param(
            "C:/user/datasets/audio/analysis/audio/audio/audio.wav",
            "/root/to/my_datasets/audio/some/stuff",
            "/root/to/my_datasets/audio/analysis/audio/audio/audio.wav",
            id="absolute_windows_path_to_posix_root_with_repeated_folders",
        ),
        pytest.param(
            "audio/analysis/audio/audio/audio.wav",
            "/root/to/my_datasets/audio/some/stuff",
            "/root/to/my_datasets/audio/some/stuff/audio/analysis/audio/audio/audio.wav",
            id="relative_path_to_posix_root_with_repeated_folders",
        ),
    ],
)
def test_relative_to_absolute(target: str, root: str, expected: str) -> None:
    path: Path = relative_to_absolute(target_path=target, root_path=root)
    assert path.resolve() == Path(PureWindowsPath(expected)).resolve()


def test_relative_paths_serialization(tmp_path: Path) -> None:
    dictionary = {
        "path": str(tmp_path / "user" / "cool"),
        "folder": str(tmp_path / "user" / "cool"),
        "json": str(tmp_path / "user" / "cool"),
        "ignored": str(tmp_path / "user" / "cool"),
        "not_a_path": "hello",
        "nested_dict": {
            "path": str(tmp_path / "user" / "cool"),
            "folder": str(tmp_path / "user" / "cool"),
            "json": str(tmp_path / "user" / "cool"),
            "ignored": str(tmp_path / "user" / "cool"),
            "not_a_path": "hello",
        },
    }

    relative_to_user_folder = {
        "path": str(Path("cool")),
        "folder": str(Path("cool")),
        "json": str(Path("cool")),
        "ignored": str(tmp_path / "user" / "cool"),
        "not_a_path": "hello",
        "nested_dict": {
            "path": str(Path("cool")),
            "folder": str(Path("cool")),
            "json": str(Path("cool")),
            "ignored": str(tmp_path / "user" / "cool"),
            "not_a_path": "hello",
        },
    }

    dict_copy = dict(dictionary)

    # Absolute to relative
    set_path_reference(
        serialized_dict=dict_copy,
        root_path=tmp_path / "user",
        reference="relative",
    )
    assert dict_copy == relative_to_user_folder

    # Relative to absolute
    set_path_reference(
        serialized_dict=dict_copy,
        root_path=tmp_path / "user",
        reference="absolute",
    )
    assert dict_copy == dictionary

    # Absolute to absolute
    set_path_reference(
        serialized_dict=dict_copy,
        root_path=tmp_path / "user",
        reference="absolute",
    )
    assert dict_copy == dictionary
