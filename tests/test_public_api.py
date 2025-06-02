from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from OSmOSE.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES,
    TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
    TIMESTAMP_FORMAT_TEST_FILES,
)
from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.event import Event
from OSmOSE.core_api.instrument import Instrument
from OSmOSE.core_api.spectro_dataset import SpectroDataset
from OSmOSE.public_api.analysis import Analysis, AnalysisType
from OSmOSE.public_api.dataset import Dataset


@pytest.mark.parametrize(
    (
        "audio_files",
        "other_files",
        "timestamp_format",
        "timezone",
        "expected_audio_events",
    ),
    [
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            [],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
            ],
            id="one_audio_file",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            [],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:01"),
                    end=Timestamp("2024-01-01 12:00:02"),
                ),
            ],
            id="multiple_audio_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 2,
                "inter_file_duration": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            [],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:02"),
                    end=Timestamp("2024-01-01 12:00:03"),
                ),
            ],
            id="multiple_audio_files_with_gap",
        ),
        pytest.param(
            {
                "duration": 2,
                "sample_rate": 48_000,
                "nb_files": 3,
                "inter_file_duration": -1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            [],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:01"),
                    end=Timestamp("2024-01-01 12:00:02"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:02"),
                    end=Timestamp("2024-01-01 12:00:04"),
                ),
            ],
            id="overlap_should_be_resolved",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            ["foo.txt", "bar.png"],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
            ],
            id="other_files_in_root",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            ["foo.txt", "foo/bar.png", "foo/cool.txt"],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
            ],
            id="other_files_in_subfolders_should_maintain_tree_structure",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            ["foo.wav", "cool/bar.wav"],
            None,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:01"),
                ),
            ],
            id="non_audio_files_with_audio_extensions_are_moved_to_others",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00", tz="America/Chihuahua"),
                "datetime_format": TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
            },
            ["foo.wav", "cool/bar.wav"],
            TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
            None,
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00", tz="America/Chihuahua"),
                    end=Timestamp("2024-01-01 12:00:01", tz="America/Chihuahua"),
                ),
            ],
            id="tz-aware_files",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            ["foo.wav", "cool/bar.wav"],
            None,
            "America/Chihuahua",
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00", tz="America/Chihuahua"),
                    end=Timestamp("2024-01-01 12:00:01", tz="America/Chihuahua"),
                ),
            ],
            id="tz-naive_files_with_provided_tz",
        ),
        pytest.param(
            {
                "duration": 1,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00", tz="America/Chihuahua"),
                "datetime_format": TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
            },
            ["foo.wav", "cool/bar.wav"],
            TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
            "Indian/Kerguelen",
            [
                Event(
                    begin=Timestamp("2024-01-01 23:00:00", tz="Indian/Kerguelen"),
                    end=Timestamp("2024-01-01 23:00:01", tz="Indian/Kerguelen"),
                ),
            ],
            id="providing_tz_should_convert_tz-aware_files",
        ),
    ],
    indirect=["audio_files"],
)
def test_dataset_build(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    other_files: list[str],
    timestamp_format: str | None,
    timezone: str | None,
    expected_audio_events: list[Event],
) -> None:
    timestamp_format = (
        TIMESTAMP_FORMAT_TEST_FILES if timestamp_format is None else timestamp_format
    )

    # add other files
    for file in other_files:
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file).touch()

    files_before_build = list(tmp_path.rglob("*"))
    original_dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=timestamp_format,
        bound="files",
        timezone=timezone,
        name="original",
    )

    dataset = Dataset(
        folder=tmp_path,
        strptime_format=timestamp_format,
        timezone=timezone,
    )

    dataset.build()

    assert not any(file.exists() for file in files_before_build)

    # Expected moved original dataset
    moved_original_dataset = deepcopy(original_dataset)
    for file in moved_original_dataset.files:
        file.path = file.path.parent / "data" / "audio" / "original" / file.path.name

    # The original dataset should be moved to the correct folder
    assert moved_original_dataset == AudioDataset.from_folder(
        folder=tmp_path / "data" / "audio" / "original",
        strptime_format=timestamp_format,
        bound="files",
        timezone=timezone,
    )

    # The original dataset should be added to the public Dataset's datasets:
    assert moved_original_dataset == dataset.datasets["original"]["dataset"]

    # Check original audio data
    assert [
        Event(d.begin, d.end) for d in dataset.datasets["original"]["dataset"].data
    ] == expected_audio_events

    # Other files should be moved to an "other" folder
    assert all((tmp_path / "other" / file).exists() for file in other_files)

    # The dataset.json file in root folder should allow for deserializing the dataset
    dataset2 = Dataset.from_json(tmp_path / "dataset.json")
    assert (
        dataset2.datasets["original"]["dataset"]
        == dataset.datasets["original"]["dataset"]
    )

    # Resetting the dataset should put back all original files back
    dataset.reset()
    assert sorted(str(file) for file in tmp_path.rglob("*")) == sorted(
        str(file) for file in files_before_build
    )


@pytest.mark.parametrize(
    ("audio_files", "analysis"),
    [
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Analysis(
                analysis_type=AnalysisType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
            ),
            id="same_format_as_original",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Analysis(
                analysis_type=AnalysisType.AUDIO,
                name="cool",
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
            ),
            id="named_dataset",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Analysis(
                analysis_type=AnalysisType.AUDIO,
                name=None,
                begin=Timestamp("2024-01-01 12:00:02"),
                end=Timestamp("2024-01-01 12:00:04"),
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
            ),
            id="part_of_the_timespan",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Analysis(
                analysis_type=AnalysisType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=Timedelta(seconds=1),
                sample_rate=None,
                subtype="DOUBLE",
            ),
            id="resize_data_with_data_duration",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Analysis(
                analysis_type=AnalysisType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=24_000,
                subtype="DOUBLE",
            ),
            id="reshaping_data",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Analysis(
                analysis_type=AnalysisType.AUDIO,
                name="fun",
                begin=Timestamp("2024-01-01 12:00:01"),
                end=Timestamp("2024-01-01 12:00:04"),
                data_duration=Timedelta(seconds=0.5),
                sample_rate=24_000,
                subtype="DOUBLE",
            ),
            id="full_reshape",
        ),
    ],
    indirect=["audio_files"],
)
def test_reshape(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    analysis: Analysis,
) -> None:
    dataset = Dataset(folder=tmp_path, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
    dataset.build()
    dataset.run_analysis(
        analysis=analysis,
    )

    expected_ads = AudioDataset.from_files(
        list(dataset.origin_dataset.files),
        begin=analysis.begin,
        end=analysis.end,
        data_duration=analysis.data_duration,
        name=analysis.name,
    )
    if analysis.sample_rate is not None:
        expected_ads.sample_rate = analysis.sample_rate

    expected_ads_name = (
        analysis.name
        if analysis.name
        else f"{expected_ads.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)}"
    )

    # The new dataset should be added to the datasets property
    assert expected_ads_name in dataset.datasets
    ads = dataset.get_dataset(expected_ads_name)
    assert ads is not None
    assert type(ads) is AudioDataset

    # Test ads values
    assert all(
        np.array_equal(ad.get_value(), expected_ad.get_value())
        for ad, expected_ad in zip(
            sorted(ads.data, key=lambda ad: ad.begin),
            sorted(expected_ads.data, key=lambda ad: ad.begin),
            strict=False,
        )
    )

    # ads folder should match the ads name
    ads_folder_name = (
        analysis.name
        if analysis.name
        else f"{round(ads.data_duration.total_seconds())}_{ads.sample_rate}"
    )
    assert ads.folder.name == ads_folder_name

    # ads should be linked to the new files instead of the originals
    assert all(file not in dataset.origin_files for file in ads.files)

    # ads should be deserializable from the exported JSON file
    json_file = ads.folder / f"{expected_ads_name}.json"
    assert json_file.exists()
    deserialized_ads = AudioDataset.from_json(json_file)
    assert deserialized_ads == ads


@pytest.mark.parametrize(
    (
        "audio_files",
        "instrument",
        "analysis",
        "expected_level",
    ),
    [
        pytest.param(
            {
                "duration": 3,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 0.1,
            },
            Instrument(end_to_end_db=150),
            Analysis(
                analysis_type=AnalysisType.AUDIO | AnalysisType.SPECTROGRAM,
                name="pingu",
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-01 12:00:01"),
                data_duration=Timedelta(seconds=1.0),
                sample_rate=24_000,
                fft=ShortTimeFFT(
                    win=hamming(1024),
                    hop=100,
                    fs=24_000,
                    scale_to="magnitude",
                ),
                subtype="DOUBLE",
            ),
            130,
            id="all_parameters_without_npz",
        ),
        pytest.param(
            {
                "duration": 3,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
                "series_type": "sine",
                "sine_frequency": 1_000,
                "magnitude": 0.1,
            },
            Instrument(end_to_end_db=150),
            Analysis(
                analysis_type=AnalysisType.AUDIO
                | AnalysisType.MATRIX
                | AnalysisType.SPECTROGRAM,
                name="pingu",
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-01 12:00:01"),
                data_duration=Timedelta(seconds=1.0),
                sample_rate=24_000,
                fft=ShortTimeFFT(
                    win=hamming(1024),
                    hop=100,
                    fs=24_000,
                    scale_to="magnitude",
                ),
                subtype="DOUBLE",
            ),
            130,
            id="all_parameters_with_npz",
        ),
    ],
    indirect=["audio_files"],
)
def test_serialization(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    instrument: Instrument | None,
    analysis: Analysis,
    expected_level: float | None,
) -> None:
    dataset = Dataset(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
        instrument=instrument,
    )
    dataset.build()
    dataset.run_analysis(
        analysis=analysis,
    )
    _, request = audio_files
    sine_frequency = request.param["sine_frequency"]

    assert analysis.name in dataset.datasets

    if AnalysisType.AUDIO in analysis.analysis_type and analysis.is_spectro:
        assert f"{analysis.name}_audio" in dataset.datasets

    if analysis.is_spectro:
        bin_idx = min(
            enumerate(analysis.fft.f),
            key=lambda t: abs(t[1] - sine_frequency),
        )[0]
        sd = dataset.get_dataset(analysis.name).data[0]
        level_tolerance = 8
        equalized_sx = sd.to_db(sd.get_value())
        computed_level = equalized_sx[bin_idx, :].mean()
        assert abs(computed_level - expected_level) < level_tolerance

    deserialized = Dataset.from_json(tmp_path / "dataset.json")

    for (t_o, d_o), (t_d, d_d) in list(
        zip(
            sorted(dataset.datasets.items(), key=lambda d: d[0]),
            sorted(deserialized.datasets.items(), key=lambda d: d[0]),
            strict=False,
        ),
    ):
        # Same analysis dataset type
        assert t_o == t_d

        if t_o is AudioDataset:
            assert np.array_equal(
                d_o.data[0].get_value_calibrated(),
                d_d.data[0].get_value_calibrated(),
            )
        if t_o is SpectroDataset:
            assert np.array_equal(
                d_o.data[0].to_db(d_o.data[0].get_value()),
                d_d.data[0].to_db(d_d.data[0].get_value()),
            )


@pytest.mark.parametrize(
    "analysis_type",
    [
        pytest.param(
            AnalysisType.SPECTROGRAM,
            id="spectro_only",
        ),
        pytest.param(
            AnalysisType.SPECTROGRAM,
            id="matrix_only",
        ),
        pytest.param(
            AnalysisType.SPECTROGRAM,
            id="both_spectral_flags",
        ),
    ],
)
def test_spectral_analysis_error_if_no_provided_fft(analysis_type: Analysis) -> None:
    with pytest.raises(
        ValueError,
        match="FFT parameter should be given if spectra outputs are selected.",
    ) as e:
        assert (
            Analysis(
                analysis_type=AnalysisType.SPECTROGRAM,
            )
            == e
        )


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        pytest.param(
            Analysis(analysis_type=AnalysisType.AUDIO),
            False,
            id="audio_only",
        ),
        pytest.param(
            Analysis(
                analysis_type=AnalysisType.SPECTROGRAM,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="spectro_only",
        ),
        pytest.param(
            Analysis(
                analysis_type=AnalysisType.MATRIX,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="matrix_only",
        ),
        pytest.param(
            Analysis(
                analysis_type=AnalysisType.MATRIX | AnalysisType.SPECTROGRAM,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="matrix_and_spectro",
        ),
        pytest.param(
            Analysis(
                analysis_type=AnalysisType.MATRIX
                | AnalysisType.SPECTROGRAM
                | AnalysisType.AUDIO,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="all_flags",
        ),
    ],
)
def test_analysis_is_spectro(analysis: Analysis, expected: bool) -> None:
    assert analysis.is_spectro is expected
