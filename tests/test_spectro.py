from __future__ import annotations

import gc
from contextlib import nullcontext
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.dates import num2date
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    TIMESTAMP_FORMATS_EXPORTED_FILES,
)
from osekit.core_api.audio_data import AudioData
from osekit.core_api.audio_dataset import AudioDataset
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.event import Event
from osekit.core_api.frequency_scale import Scale, ScalePart
from osekit.core_api.instrument import Instrument
from osekit.core_api.ltas_data import LTASData
from osekit.core_api.ltas_dataset import LTASDataset
from osekit.core_api.spectro_data import SpectroData
from osekit.core_api.spectro_dataset import SpectroDataset
from osekit.core_api.spectro_file import SpectroFile
from osekit.utils.audio_utils import Normalization, generate_sample_audio

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
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )
    spectro_dataset = SpectroDataset.from_audio_dataset(dataset, sft)
    for audio, spectro in zip(dataset.data, spectro_dataset.data, strict=False):
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
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            pd.Timestamp("2024-01-01 12:00:00.123+0200"),
            ShortTimeFFT(hamming(512), 512, 1_024),
            id="localized_spectro",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectro_parameters_in_npz_files(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    date_begin: pd.Timestamp,
    sft: ShortTimeFFT,
) -> None:
    afs, _ = audio_files

    ad = AudioData.from_files(afs)
    sd = SpectroData.from_audio_data(ad, sft)
    sd.write(tmp_path / "npz")
    file = tmp_path / "npz" / f"{sd}.npz"
    sf = SpectroFile(file, strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES)

    assert sf.begin == ad.begin
    assert sf.end == ad.end
    assert np.array_equal(sf.freq, sft.f)
    assert sf.hop == sft.hop
    assert sf.mfft == sft.mfft
    assert sf.sample_rate == sft.fs
    nb_time_bins = sft.t(ad.shape).shape[0]
    assert np.array_equal(
        sf.time,
        np.arange(nb_time_bins) * ad.duration.total_seconds() / nb_time_bins,
    )


@pytest.mark.parametrize(
    ("audio_files", "instrument", "normalization", "nb_chunks", "sft"),
    [
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
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
            None,
            None,
            6,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="6_seconds_split_in_6_with_overlap",
        ),
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Instrument(end_to_end_db=150.0),
            None,
            6,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="audio_data_with_instrument",
        ),
        pytest.param(
            {
                "duration": 6,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Normalization.ZSCORE,
            6,
            ShortTimeFFT(hamming(1_024), 100, 1_024),
            id="audio_data_with_normalization",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectrogram_from_npz_files(
    tmp_path: Path,
    audio_files: tuple[list[Path], pytest.fixtures.Subrequest],
    instrument: Instrument | None,
    normalization: Normalization | None,
    nb_chunks: int,
    sft: ShortTimeFFT,
) -> None:
    afs = [
        AudioFile(f, strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED)
        for f in tmp_path.glob("*.wav")
    ]

    ad = AudioData.from_files(files=afs, instrument=instrument)
    if normalization:
        ad.normalization = normalization
    sd = SpectroData.from_audio_data(ad, sft)

    sd_split = sd.split(nb_chunks)

    import soundfile as sf

    for spectro in sd_split:
        spectro.write(tmp_path / "output")
        centered_data = spectro.audio_data.get_value()
        (tmp_path / "audio").mkdir(exist_ok=True)
        sf.write(
            file=tmp_path / "audio" / f"{spectro.audio_data}.wav",
            data=centered_data,
            samplerate=spectro.audio_data.sample_rate,
            subtype="DOUBLE",
        )

    assert len(list((tmp_path / "output").glob("*.npz"))) == nb_chunks

    afs = [
        AudioFile(f, strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED)
        for f in (tmp_path / "audio").glob("*.wav")
    ]
    ad = AudioData.from_files(afs, instrument=instrument)
    if normalization:
        ad.normalization = normalization
    sd = SpectroData.from_audio_data(ad, sft)

    sds = SpectroDataset.from_folder(
        tmp_path / "output",
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    assert sds.begin == ad.begin

    # Beats me, but Timedelta.round() raises a DivideByZeroException if done
    # directly on the duration properties.

    dt1, dt2 = (Timedelta(str(dt)) for dt in (sds.duration, ad.duration))

    assert dt1.round(freq="ms") == dt2.round(freq="ms")
    assert len(sds.data) == 1
    assert sds.data[0].shape == sds.data[0].get_value().shape

    assert sds.data[0].shape == (
        sft.f.shape[0],
        sft.p_num(int(ad.duration.total_seconds() * ad.sample_rate)),
    )

    assert np.allclose(sd.get_value(), sds.data[0].get_value())


@pytest.mark.parametrize(
    ("audio_files", "origin_dtype", "target_dtype", "expected_value_dtype"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            complex,
            complex,
            complex,
            id="complex_to_complex",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            complex,
            float,
            float,
            id="complex_to_float",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            complex,
            float,
            float,
            id="float_to_float",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            float,
            complex,
            pytest.raises(
                TypeError,
                match="Cannot convert absolute npz values to complex sx values.",
            ),
            id="float_to_complex_raises_exception",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectrogram_sx_dtype(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    origin_dtype: type[complex],
    target_dtype: type[complex],
    expected_value_dtype: type[complex],
) -> None:
    audio_files, request = audio_files
    ad = AudioData.from_files(audio_files)
    sft = ShortTimeFFT(hamming(128), 128, 1_024)
    sd = SpectroData.from_audio_data(ad, sft)

    sd.sx_dtype = origin_dtype
    ltas = LTASData.from_spectro_data(sd, 4)
    assert ltas.sx_dtype is float  # Default LTASData behaviour

    assert sd.get_value().dtype == origin_dtype

    sd.write(tmp_path / "npz")

    sf = SpectroFile(
        tmp_path / "npz" / f"{sd}.npz",
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    assert sf.sx_dtype is origin_dtype

    sd2 = SpectroData.from_files([sf])

    assert sd2.sx_dtype is origin_dtype  # Default SpectroData behaviour

    assert ltas.get_value().dtype == float

    sd2.sx_dtype = target_dtype

    if type(expected_value_dtype) is type:
        assert sd2.get_value().dtype == expected_value_dtype
    else:
        with expected_value_dtype:
            assert sd2.get_value().dtype == expected_value_dtype

    sd2.sx_dtype = origin_dtype

    assert sd2.get_value().dtype == origin_dtype


@pytest.mark.parametrize(
    ("audio_files", "ad1", "ad1_sr", "ad2", "ad2_sr", "expected_exception"),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:01"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:01"),
            ),
            1_024,
            nullcontext(),
            id="equal_audio_data",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00+0200"),
                end=pd.Timestamp("2024-01-01 12:00:01+0200"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00+0200"),
                end=pd.Timestamp("2024-01-01 12:00:01+0200"),
            ),
            1_024,
            nullcontext(),
            id="equal_localized_audio_data",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00+0200"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00+0200"),
                end=pd.Timestamp("2024-01-01 12:00:01+0200"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00+0000"),
                end=pd.Timestamp("2024-01-01 12:00:01+0000"),
            ),
            1_024,
            pytest.raises(
                ValueError,
                match="The begin of the audio data doesn't match.",
            ),
            id="different_timezones",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:02"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:01"),
                end=pd.Timestamp("2024-01-01 12:00:02"),
            ),
            1_024,
            pytest.raises(
                ValueError,
                match="The begin of the audio data doesn't match.",
            ),
            id="different_begin",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:02"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:01"),
            ),
            1_024,
            pytest.raises(ValueError, match="The end of the audio data doesn't match."),
            id="different_end",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:01"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:01"),
            ),
            2_048,
            pytest.raises(
                ValueError,
                match="The sample rate of the audio data doesn't match.",
            ),
            id="different_sample_rate",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:00"),
                end=pd.Timestamp("2024-01-01 12:00:01"),
            ),
            1_024,
            Event(
                begin=pd.Timestamp("2024-01-01 12:00:01"),
                end=pd.Timestamp("2024-01-01 12:00:02"),
            ),
            1_024,
            pytest.raises(
                ValueError,
                match="The begin of the audio data doesn't match.",
            ),
            id="different_timespan",
        ),
    ],
    indirect=["audio_files"],
)
def test_link_audio_data(
    audio_files: pytest.fixture,
    tmp_path: Path,
    ad1: Event,
    ad1_sr: float,
    ad2: Event,
    ad2_sr: float,
    expected_exception: type[Exception],
) -> None:
    audio_files, request = audio_files

    ad1 = AudioData.from_files(audio_files, begin=ad1.begin, end=ad1.end)
    ad1.sample_rate = ad1_sr

    ad2 = AudioData.from_files(audio_files, begin=ad2.begin, end=ad2.end)
    ad2.sample_rate = ad2_sr

    sd = SpectroData.from_audio_data(
        ad1,
        ShortTimeFFT(hamming(128), 128, ad1.sample_rate),
    )

    assert sd.audio_data is ad1
    assert sd.audio_data is not ad2

    with expected_exception as e:
        assert sd.link_audio_data(ad2) == e

    if type(expected_exception) is not nullcontext:
        return

    assert sd.audio_data is not ad1
    assert sd.audio_data is ad2


@pytest.mark.parametrize(
    (
        "audio_files",
        "ads1_data_duration",
        "ads2_data_duration",
        "ads2_sample_rate",
        "start_index",
        "stop_index",
        "expected_exception",
    ),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timedelta(seconds=0.1),
            Timedelta(seconds=0.1),
            1_024,
            None,
            None,
            nullcontext(),
            id="default_indexes_is_full_dataset",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timedelta(seconds=0.1),
            Timedelta(seconds=0.1),
            1_024,
            2,
            6,
            nullcontext(),
            id="link_a_part_of_the_data",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timedelta(seconds=0.1),
            Timedelta(seconds=0.1),
            2_048,
            None,
            None,
            pytest.raises(
                ValueError,
                match="The sample rate of the audio data doesn't match.",
            ),
            id="different_sample_rate",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timedelta(seconds=0.1),
            Timedelta(seconds=0.5),
            1_024,
            None,
            None,
            pytest.raises(
                ValueError,
                match="The audio dataset doesn't contain the same number of data as the"
                " spectro dataset.",
            ),
            id="different_number_of_data",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 1_024,
                "nb_files": 1,
                "date_begin": pd.Timestamp("2024-01-01 12:00:00"),
            },
            Timedelta(seconds=0.1),
            Timedelta(seconds=0.101),
            1_024,
            None,
            None,
            pytest.raises(ValueError, match="The end of the audio data doesn't match."),
            id="different_end_of_first_data",
        ),
    ],
    indirect=["audio_files"],
)
def test_link_audio_dataset(
    audio_files: pytest.fixture,
    tmp_path: pytest.fixture,
    ads1_data_duration: Timedelta,
    ads2_data_duration: Timedelta,
    ads2_sample_rate: float,
    start_index: int,
    stop_index: int,
    expected_exception: type[Exception],
) -> None:
    ads1 = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        data_duration=ads1_data_duration,
    )
    ads2 = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        data_duration=ads2_data_duration,
    )
    ads2.sample_rate = ads2_sample_rate

    sds = SpectroDataset.from_audio_dataset(
        ads1,
        fft=ShortTimeFFT(hamming(128), 128, ads1.sample_rate),
    )

    with expected_exception as e:
        assert sds.link_audio_dataset(ads2, first=start_index, last=stop_index) == e

    if type(expected_exception) is not nullcontext:
        return

    start_index = 0 if start_index is None else start_index
    stop_index = len(ads1.data) if stop_index is None else stop_index

    for idx, sd in enumerate(sds.data):
        if idx in range(start_index, stop_index):
            assert sd.audio_data is not ads1.data[idx]
            assert sd.audio_data is ads2.data[idx]
        else:
            assert sd.audio_data is ads1.data[idx]
            assert sd.audio_data is not ads2.data[idx]

    # linking should fail if the length of the audio datasets differ:
    ads_err = AudioDataset(
        [*ads2.data, ads2.data[0]],
    )  # Adding one data to the destination ads
    with pytest.raises(
        ValueError,
        match="The audio dataset doesn't contain the same number of data as the "
        "spectro dataset.",
    ) as exc_info:
        assert sds.link_audio_dataset(ads_err) == exc_info

    # linking should fail if any of the data can't be linked
    ads_err = AudioDataset(ads1.data)
    ads1.data[-1].sample_rate = ads2_sample_rate * 0.5
    with pytest.raises(
        ValueError,
        match="The sample rate of the audio data doesn't match.",
    ):
        assert sds.link_audio_dataset(ads_err) == exc_info


@pytest.mark.parametrize(
    ("v_lim1", "v_lim2"),
    [
        pytest.param(
            None,
            (0.0, 150.0),
            id="none_to_tuple",
        ),
        pytest.param(
            (0.0, 150.0),
            (-120.0, 0.0),
            id="tuple_to_tuple",
        ),
    ],
)
def test_spectrodataset_vlim(
    audio_files: pytest.fixture,
    tmp_path: pytest.fixture,
    v_lim1: tuple[float, float] | None,
    v_lim2: tuple[float, float] | None,
) -> None:
    afs, _ = audio_files
    sft = ShortTimeFFT(hamming(128), 128, afs[0].sample_rate)
    ads = AudioDataset.from_files(afs, data_duration=afs[0].duration / 10)

    if v_lim1 is None:
        sds = SpectroDataset.from_audio_dataset(
            ads,
            sft,
        )
    else:
        sds = SpectroDataset.from_audio_dataset(
            ads,
            sft,
            v_lim=v_lim1,
        )

    default_v_lim = SpectroData.from_audio_data(ads.data[0], sft).v_lim
    if v_lim1 is None:
        v_lim1 = default_v_lim

    assert all(sd.v_lim == v_lim1 for sd in sds.data)

    sds.v_lim = v_lim2

    assert all(sd.v_lim == v_lim2 for sd in sds.data)

    sds.write_json(tmp_path / "json")

    sds2 = SpectroDataset.from_json(next((tmp_path / "json").iterdir()))

    assert all(sd.v_lim == v_lim2 for sd in sds2.data)

    # Different v_lim in the SpectroDataset
    sd1, sd2 = sds.data[:2]

    sd1.v_lim = v_lim1
    sd2.v_lim = v_lim2

    sds3 = SpectroDataset([sd1, sd2])

    assert sds3.data[0].v_lim == v_lim1
    assert sds3.data[1].v_lim == v_lim2

    sds3.v_lim = None

    assert sds3.data[0].v_lim == default_v_lim
    assert sds3.data[1].v_lim == default_v_lim


@pytest.mark.parametrize(
    ("audio_files", "sft", "parts", "v_lim", "colormap"),
    [
        pytest.param(
            {},
            ShortTimeFFT(
                hamming(1024),
                1024,
                48_000,
            ),
            1,
            None,
            None,
            id="default_parameters_one_subdata",
        ),
        pytest.param(
            {},
            ShortTimeFFT(
                hamming(1024),
                1024,
                48_000,
            ),
            5,
            None,
            None,
            id="default_parameters_5_subdata",
        ),
        pytest.param(
            {},
            ShortTimeFFT(
                hamming(1024),
                1024,
                48_000,
            ),
            5,
            (0.0, 150.0),
            None,
            id="specified_v_lim",
        ),
        pytest.param(
            {},
            ShortTimeFFT(
                hamming(1024),
                1024,
                48_000,
            ),
            5,
            None,
            "inferno",
            id="specified_colormap",
        ),
        pytest.param(
            {},
            ShortTimeFFT(
                hamming(1024),
                1024,
                48_000,
            ),
            5,
            (0.0, 150.0),
            "inferno",
            id="specified_all",
        ),
    ],
    indirect=["audio_files"],
)
def test_spectrodata_split(
    audio_files: pytest.fixture,
    sft: ShortTimeFFT,
    parts: int,
    v_lim: tuple[float, float],
    colormap: str,
) -> None:
    files, _ = audio_files
    ad = AudioData.from_files(files)
    sd = SpectroData.from_audio_data(
        data=ad,
        fft=sft,
        v_lim=v_lim,
        colormap=colormap,
    )
    sd_parts = sd.split(parts)
    for sd_part in sd_parts:
        assert sd_part.fft is sd.fft
        assert sd_part.v_lim == sd.v_lim
        assert sd_part.colormap == sd.colormap
    assert sd_parts[0].begin == sd.begin
    assert sd_parts[-1].end == sd.end


def test_ltas(audio_files: pytest.fixture, tmp_path: pytest.fixture) -> None:
    audio_files, _ = audio_files
    ad = AudioData.from_files(audio_files)
    sd = SpectroData.from_audio_data(
        data=ad,
        fft=ShortTimeFFT(hamming(1024), 512, ad.sample_rate),
    )

    nb_time_bins = 4
    ltas = LTASData.from_spectro_data(spectro_data=sd, nb_time_bins=nb_time_bins)

    assert ltas.fft.hop == ltas.fft.win.shape[0]
    assert ltas.shape == (sd.shape[0], nb_time_bins)
    sx = ltas.get_value()
    assert sx.shape == ltas.shape

    ltas2 = LTASData.from_dict(dictionary=ltas.to_dict())
    assert np.array_equal(sx, ltas2.get_value())

    ltas_ds = LTASDataset([ltas])
    ltas_ds.write_json(tmp_path)
    ltas_ds2 = LTASDataset.from_json(tmp_path / f"{ltas_ds.name}.json")

    assert type(ltas_ds2) is LTASDataset
    assert type(ltas_ds2.data[0]) is LTASData
    assert np.array_equal(ltas_ds.data[0].get_value(), ltas_ds2.data[0].get_value())


@pytest.mark.parametrize(
    "audio_files",
    [{"date_begin": Timestamp("2020-01-01 00:00:00", tz="UTC")}],
    indirect=True,
)
def test_spectro_axis(
    audio_files: tuple[list[AudioFile], ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audio_files, _ = audio_files
    ad = AudioData.from_files(audio_files)
    sd = SpectroData.from_audio_data(
        data=ad,
        fft=ShortTimeFFT(hamming(1024), 512, ad.sample_rate),
    )

    plot_kwargs = {}

    def mock_imshow(
        self: plt.Axes,
        sx: np.ndarray,
        **kwargs: str,
    ) -> None:
        plot_kwargs["sx"] = sx

        for k, v in kwargs.items():
            plot_kwargs[k] = v

    monkeypatch.setattr(plt.Axes, "imshow", mock_imshow)

    sd.plot()

    assert (plot_kwargs["vmin"], plot_kwargs["vmax"]) == sd.v_lim
    assert plot_kwargs["cmap"] == sd.colormap
    assert plot_kwargs["origin"] == "lower"
    assert plot_kwargs["aspect"] == "auto"
    assert plot_kwargs["interpolation"] == "none"

    t1, t2, f1, f2 = plot_kwargs["extent"]
    t1, t2 = map(num2date, (t1, t2))
    t1, t2 = map(Timestamp, (t1, t2))
    assert t1 == sd.begin
    assert t2 == sd.end
    assert f1 == sd.fft.f[0]
    assert f2 == sd.fft.f[-1]


def test_spectro_default_v_lim(audio_files: pytest.fixture) -> None:
    files, _ = audio_files
    ad = AudioData.from_files(files)
    ad_inst = AudioData.from_files(files, instrument=Instrument(end_to_end_db=150.0))

    sft = ShortTimeFFT(win=hamming(1024), hop=128, fs=ad.sample_rate)

    sd = SpectroData.from_audio_data(ad, sft)
    sd_inst = SpectroData.from_audio_data(ad_inst, sft)

    assert sd.v_lim == (-120.0, 0.0)
    assert sd_inst.v_lim == (0.0, 170.0)


def test_garbage_collection_after_save_spectrogram(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], pytest.fixtures.Subrequest],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files, _ = audio_files
    ad = AudioData.from_files(files)
    sft = ShortTimeFFT(win=hamming(1024), hop=128, fs=ad.sample_rate)

    sd = SpectroData.from_audio_data(ad, sft)
    ltas = LTASData.from_spectro_data(spectro_data=sd, nb_time_bins=4)

    collect_calls = [0]

    def patch_collect() -> None:
        collect_calls[0] += 1

    monkeypatch.setattr(gc, "collect", patch_collect)
    monkeypatch.setattr(SpectroData, "plot", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "savefig", lambda *args, **kwargs: None)

    sd.save_spectrogram(tmp_path / "output")

    assert collect_calls[0] == 1

    ltas.save_spectrogram(tmp_path / "output")

    assert collect_calls[0] == 2  # noqa: PLR2004

    sds = SpectroDataset([sd] * 5)
    sds.save_spectrogram(tmp_path / "output")

    assert collect_calls[0] == 7  # noqa: PLR2004

    ltass = LTASDataset([ltas] * 5)
    ltass.save_spectrogram(tmp_path / "output")

    assert collect_calls[0] == 12  # noqa: PLR2004


def test_spectrodataset_scale(
    tmp_path: Path,
    patch_audio_data: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ad = AudioData(
        mocked_value=np.linspace(0.0, 1.0, 1000),  # Type: ignore # Unexpected argument
    )

    fft = ShortTimeFFT(win=hamming(512), hop=128, fs=ad.sample_rate)
    scale = Scale(
        [
            ScalePart(0.0, 0.25, 0.0, 500.0),
            ScalePart(0.25, 1.0, 500.0, 20_000.0, scale_type="log"),
        ],
    )

    sd = SpectroData.from_audio_data(data=ad, fft=fft)

    sds = SpectroDataset(data=[sd], scale=scale)

    def mock_plot(*args: list[...], **kwargs: dict) -> None:
        assert kwargs["scale"] == scale

    def mock_empty_method(*args: list[...], **kwargs: dict) -> None:
        return

    monkeypatch.setattr(SpectroData, "plot", mock_plot)
    monkeypatch.setattr(SpectroData, "write", mock_empty_method)
    monkeypatch.setattr(plt, "savefig", mock_empty_method)
    sds.save_spectrogram(tmp_path)
    sds.save_all(tmp_path, tmp_path)

    ltas = LTASData.from_spectro_data(sd, nb_time_bins=50)
    ltas_ds = LTASDataset([ltas])
    ltas_ds.scale = scale

    ltas_ds.save_spectrogram(tmp_path)
    ltas_ds.save_all(tmp_path, tmp_path)
