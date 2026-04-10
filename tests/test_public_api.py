from __future__ import annotations

import itertools
from contextlib import AbstractContextManager, nullcontext
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import hamming

from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
)
from osekit.core.audio_dataset import AudioDataset
from osekit.core.audio_file import AudioFile
from osekit.core.event import Event
from osekit.core.frequency_scale import Scale, ScalePart
from osekit.core.instrument import Instrument
from osekit.core.ltas_dataset import LTASDataset
from osekit.core.spectro_dataset import SpectroDataset
from osekit.public.project import Project
from osekit.public.transform import OutputType, Transform
from osekit.utils.audio import Normalization


@pytest.mark.parametrize(
    (
        "audio_files",
        "other_files",
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
            },
            ["foo.wav", "cool/bar.wav"],
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
            },
            ["foo.wav", "cool/bar.wav"],
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
def test_project_build(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    other_files: list[str],
    expected_audio_events: list[Event],
) -> None:
    _, request = audio_files
    file_timezone = (
        None
        if request.param["date_begin"].tzinfo is None
        else str(request.param["date_begin"])
    )
    project_timezone = (
        None
        if expected_audio_events[0].begin.tzinfo is None
        else str(expected_audio_events[0].begin.tzinfo)
    )

    files_timestamp_format = (
        TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
        if file_timezone is None
        else TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED
    )

    # add other files
    for file in other_files:
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file).touch()

    files_before_build = list(tmp_path.rglob("*"))
    original_dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=files_timestamp_format,
        mode="files",
        timezone=project_timezone,
        name="original",
    )

    project = Project(
        folder=tmp_path,
        strptime_format=files_timestamp_format,
        timezone=project_timezone,
    )

    project.build()

    assert not any(file.exists() for file in files_before_build)

    # Expected moved original dataset
    moved_original_dataset = deepcopy(original_dataset)
    for file in moved_original_dataset.files:
        file.path = file.path.parent / "data" / "audio" / "original" / file.path.name

    # The original dataset should be moved to the correct folder
    assert moved_original_dataset == AudioDataset.from_folder(
        folder=tmp_path / "data" / "audio" / "original",
        strptime_format=files_timestamp_format,
        mode="files",
        timezone=project_timezone,
    )

    # The original dataset should be added to the public Project's output datasets:
    assert moved_original_dataset == project.outputs["original"]["dataset"]

    # Check original audio data
    assert [
        Event(d.begin, d.end) for d in project.outputs["original"]["dataset"].data
    ] == expected_audio_events

    # Other files should be moved to an "other" folder
    assert all((tmp_path / "other" / file).exists() for file in other_files)

    # The project.json file in root folder should allow for deserializing the dataset
    project2 = Project.from_json(tmp_path / "project.json")
    assert project2.origin_dataset == project.outputs["original"]["dataset"]

    # Resetting the project should put back all original files back
    project.reset()
    assert sorted(str(file) for file in tmp_path.rglob("*")) == sorted(
        str(file) for file in files_before_build
    )


@pytest.mark.parametrize(
    ("audio_files", "transform"),
    [
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Transform(
                output_type=OutputType.AUDIO,
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
            Transform(
                output_type=OutputType.AUDIO,
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
            Transform(
                output_type=OutputType.AUDIO,
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
            Transform(
                output_type=OutputType.AUDIO,
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
            Transform(
                output_type=OutputType.AUDIO,
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
            Transform(
                output_type=OutputType.AUDIO,
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
    transform: Transform,
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )
    project.build()
    project.run(
        transform=transform,
    )

    expected_ads = AudioDataset.from_files(
        list(project.origin_dataset.files),
        begin=transform.begin,
        end=transform.end,
        data_duration=transform.data_duration,
        name=transform.name,
    )
    if transform.sample_rate is not None:
        expected_ads.sample_rate = transform.sample_rate

    expected_ads_name = (
        transform.name
        if transform.name
        else f"{expected_ads.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED)}"
    )

    # The new dataset should be added to the outputs property
    assert expected_ads_name in project.outputs
    ads = project.get_output(expected_ads_name)
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
        transform.name
        if transform.name
        else f"{round(ads.data_duration.total_seconds())}_{ads.sample_rate}"
    )
    assert ads.folder.name == ads_folder_name

    # ads should be linked to the new files instead of the originals
    assert all(file not in project.origin_files for file in ads.files)

    # ads should be deserializable from the exported JSON file
    json_file = ads.folder / f"{expected_ads_name}.json"
    assert json_file.exists()
    deserialized_ads = AudioDataset.from_json(json_file)
    assert deserialized_ads == ads


@pytest.mark.parametrize(
    (
        "audio_files",
        "instrument",
        "transform",
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
            Transform(
                output_type=OutputType.AUDIO | OutputType.SPECTROGRAM,
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
            Transform(
                output_type=OutputType.AUDIO
                | OutputType.SPECTRUM
                | OutputType.SPECTROGRAM,
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
    transform: Transform,
    expected_level: float | None,
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        instrument=instrument,
    )
    project.build()
    project.run(
        transform=transform,
    )
    _, request = audio_files
    sine_frequency = request.param["sine_frequency"]

    assert transform.name in project.outputs

    if OutputType.AUDIO in transform.output_type and transform.is_spectro:
        assert f"{transform.name}_audio" in project.outputs

    if transform.is_spectro:
        bin_idx = min(
            enumerate(transform.fft.f),
            key=lambda t: abs(t[1] - sine_frequency),
        )[0]
        sd = project.get_output(transform.name).data[0]
        level_tolerance = 8
        equalized_sx = sd._to_db(sd.get_value())
        computed_level = equalized_sx[bin_idx, :].mean()
        assert abs(computed_level - expected_level) < level_tolerance

    deserialized = Project.from_json(tmp_path / "project.json")

    # transform dataset deserialization is only done on request
    assert all(
        isinstance(output_dataset["dataset"], Path)
        for output_dataset in deserialized.outputs.values()
    )

    for (t_o, d_o), (t_d, d_d) in zip(
        sorted(project.outputs.items(), key=lambda d: d[0]),
        sorted(deserialized.outputs.items(), key=lambda d: d[0]),
        strict=False,
    ):
        # Same transform dataset type
        assert t_o == t_d

        if t_o is AudioDataset:
            assert np.array_equal(
                d_o.data[0].get_value_calibrated(),
                d_d.data[0].get_value_calibrated(),
            )
        if t_o is SpectroDataset:
            assert np.array_equal(
                d_o.data[0]._to_db(d_o.data[0].get_value()),
                d_d.data[0]._to_db(d_d.data[0].get_value()),
            )


@pytest.mark.parametrize(
    "output_type",
    [
        pytest.param(
            OutputType.SPECTROGRAM,
            id="spectro_only",
        ),
        pytest.param(
            OutputType.SPECTRUM,
            id="spectrum_only",
        ),
        pytest.param(
            OutputType.SPECTRUM | OutputType.SPECTROGRAM,
            id="both_spectral_flags",
        ),
    ],
)
def test_spectral_transform_error_if_no_provided_fft(output_type: OutputType) -> None:
    with pytest.raises(
        ValueError,
        match=r"FFT parameter should be given if spectra outputs are selected.",
    ):
        Transform(
            output_type=OutputType.SPECTROGRAM,
        )


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        pytest.param(
            Transform(output_type=OutputType.AUDIO),
            False,
            id="audio_only",
        ),
        pytest.param(
            Transform(
                output_type=OutputType.SPECTROGRAM,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="spectro_only",
        ),
        pytest.param(
            Transform(
                output_type=OutputType.SPECTRUM,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="spectrum_only",
        ),
        pytest.param(
            Transform(
                output_type=OutputType.SPECTRUM | OutputType.SPECTROGRAM,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="spectrum_and_spectro",
        ),
        pytest.param(
            Transform(
                output_type=OutputType.SPECTRUM
                | OutputType.SPECTROGRAM
                | OutputType.AUDIO,
                fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
            ),
            True,
            id="all_flags",
        ),
    ],
)
def test_transform_is_spectro(transform: Transform, expected: bool) -> None:
    assert transform.is_spectro is expected


def test_transform_constructor_rejects_mismatched_fs() -> None:
    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        Transform(
            output_type=OutputType.SPECTROGRAM,
            sample_rate=32_000,
            fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
        )


def test_transform_rejects_setting_fft_with_wrong_fs() -> None:
    transform = Transform(
        output_type=OutputType.SPECTROGRAM,
        sample_rate=48_000,
        fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
    )

    with pytest.raises(
        ValueError,
        match="does not match",
    ):
        transform.fft = ShortTimeFFT(hamming(1024), 1024, 32_000)


def test_transform_sample_rate_propagates_to_fft() -> None:
    transform = Transform(
        output_type=OutputType.SPECTROGRAM,
        sample_rate=48_000,
        fft=ShortTimeFFT(hamming(1024), 1024, 48_000),
    )

    new_samplerate = 32_000
    transform.sample_rate = new_samplerate

    assert transform.sample_rate == new_samplerate
    assert transform.fft.fs == new_samplerate


@pytest.mark.parametrize(
    ("sample_rate", "fft", "expected"),
    [
        pytest.param(
            None,
            None,
            nullcontext(),
            id="no_sample_rate_nor_fft_shouldnt_raise",
        ),
        pytest.param(
            None,
            ShortTimeFFT(hamming(1024), 1024, 48_000),
            nullcontext(),
            id="no_sample_rate_shouldnt_raise",
        ),
        pytest.param(
            48_000,
            None,
            nullcontext(),
            id="no_fft_shouldnt_raise",
        ),
        pytest.param(
            48_000,
            ShortTimeFFT(hamming(1024), 1024, 48_000),
            nullcontext(),
            id="matching_sample_rate_and_fft_shouldnt_raise",
        ),
        pytest.param(
            32_000,
            ShortTimeFFT(hamming(1024), 1024, 48_000),
            pytest.raises(ValueError, match="does not match"),
            id="mismatching_sample_rate_and_fft_raises",
        ),
    ],
)
def test_transform_validate_sample_rate(
    sample_rate: float | None,
    fft: ShortTimeFFT | None,
    expected: AbstractContextManager,
) -> None:
    with expected:
        Transform(OutputType.AUDIO)._validate_sample_rate(
            sample_rate=sample_rate,
            fft=fft,
        )


@pytest.mark.parametrize(
    ("audio_files", "instrument", "transform", "expected_data"),
    [
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="only_one_data",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Instrument(end_to_end_db=150),
            Transform(
                output_type=OutputType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="ads_has_project_instrument",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=Timedelta(seconds=1),
                sample_rate=None,
                subtype="DOUBLE",
            ),
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
                    end=Timestamp("2024-01-01 12:00:03"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:03"),
                    end=Timestamp("2024-01-01 12:00:04"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:04"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="reshaped_ads",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=24_000,
                subtype="DOUBLE",
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="resampled_ads",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.AUDIO,
                name="cool",
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="named_ads",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.SPECTROGRAM,
                name="cool",
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                subtype="DOUBLE",
                fft=ShortTimeFFT(hamming(1024), 512, 24_000),
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="named_ads_in_spectro_transform",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.AUDIO,
                name=None,
                begin=Timestamp("2024-01-01 12:00:02"),
                end=Timestamp("2024-01-01 12:00:04"),
                data_duration=Timedelta(seconds=1),
                sample_rate=None,
                subtype="DOUBLE",
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:02"),
                    end=Timestamp("2024-01-01 12:00:03"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:03"),
                    end=Timestamp("2024-01-01 12:00:04"),
                ),
            ],
            id="specified_begin_and_end",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Instrument(end_to_end_db=150),
            Transform(
                output_type=OutputType.SPECTROGRAM,
                name="cool",
                begin=Timestamp("2024-01-01 12:00:02"),
                end=Timestamp("2024-01-01 12:00:04"),
                data_duration=Timedelta(seconds=1),
                sample_rate=24_000,
                subtype="DOUBLE",
                fft=ShortTimeFFT(hamming(1024), 512, 24_000),
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:02"),
                    end=Timestamp("2024-01-01 12:00:03"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:03"),
                    end=Timestamp("2024-01-01 12:00:04"),
                ),
            ],
            id="full_reshape",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Transform(
                output_type=OutputType.AUDIO,
                name=None,
                begin=None,
                end=None,
                data_duration=None,
                sample_rate=None,
                normalization="zscore",
                subtype="DOUBLE",
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-01 12:00:05"),
                ),
            ],
            id="normalized_data",
        ),
    ],
    indirect=["audio_files"],
)
def test_prepare_audio(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    instrument: Instrument | None,
    transform: Transform,
    expected_data: list[Event],
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        instrument=instrument,
    )
    project.build()

    transform_ds = project.prepare_audio(transform=transform)

    assert all(
        ad.begin == e.begin and ad.end == e.end
        for ad, e in zip(
            sorted(transform_ds.data, key=lambda e: e.begin),
            sorted(expected_data, key=lambda e: e.begin),
            strict=True,
        )
    )

    if transform.name is not None:
        assert str(transform_ds) == transform.name + (
            "_audio" if transform.is_spectro else ""
        )
    assert (
        transform_ds.sample_rate == project.origin_dataset.sample_rate
        if transform.sample_rate is None
        else transform.sample_rate
    )

    assert transform_ds.instrument is project.instrument

    assert transform_ds.normalization == transform.normalization


@pytest.mark.parametrize(
    ("audio_files", "instrument", "transform", "expected_data"),
    [
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            Instrument(end_to_end_db=150),
            Transform(
                output_type=OutputType.SPECTROGRAM,
                name="cool",
                begin=Timestamp("2024-01-01 12:00:02"),
                end=Timestamp("2024-01-01 12:00:04"),
                data_duration=Timedelta(seconds=1),
                sample_rate=24_000,
                subtype="DOUBLE",
                fft=ShortTimeFFT(hamming(1024), 512, 24_000),
                scale=Scale(
                    [
                        ScalePart(0.0, 0.5, 0.0, 24_000, "lin"),
                        ScalePart(0.0, 0.5, 1000.0, 24_000, "log"),
                    ],
                ),
            ),
            [
                Event(
                    begin=Timestamp("2024-01-01 12:00:02"),
                    end=Timestamp("2024-01-01 12:00:03"),
                ),
                Event(
                    begin=Timestamp("2024-01-01 12:00:03"),
                    end=Timestamp("2024-01-01 12:00:04"),
                ),
            ],
            id="full_transform",
        ),
    ],
    indirect=["audio_files"],
)
def test_prepare_spectro(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    instrument: Instrument | None,
    transform: Transform,
    expected_data: list[Event],
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        instrument=instrument,
    )
    project.build()

    transform_sds = project.prepare_spectro(transform=transform)

    assert all(
        ad.begin == e.begin and ad.end == e.end
        for ad, e in zip(
            sorted(transform_sds.data, key=lambda e: e.begin),
            sorted(expected_data, key=lambda e: e.begin),
            strict=True,
        )
    )

    if transform.name is not None:
        assert str(transform_sds) == transform.name

    assert (
        transform_sds.data[0].audio_data.sample_rate
        == project.origin_dataset.sample_rate
        if transform.sample_rate is None
        else transform.sample_rate
    )

    assert transform_sds.data[0].audio_data.instrument is project.instrument

    assert transform_sds.fft is transform.fft
    assert transform_sds.scale is transform.scale

    # FFT should be provided for spectral outputs
    with pytest.raises(
        ValueError,
        match=r"FFT parameter should be given if spectra outputs are selected.",
    ):
        project.prepare_spectro(
            transform=Transform(
                output_type=OutputType.SPECTROGRAM,
            ),
        )

    # transform.nb_ltas_time_bins implies LTASDataset output
    transform.nb_ltas_time_bins = 200
    assert type(project.prepare_spectro(transform=transform)) is LTASDataset


def test_edit_transform_before_run(
    tmp_path: Path,
    audio_files: None,
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
        instrument=Instrument(end_to_end_db=150),
    )

    project.build()

    transform = Transform(
        output_type=OutputType.AUDIO | OutputType.SPECTRUM | OutputType.SPECTROGRAM,
        data_duration=project.origin_dataset.duration / 2,
        name="original_transform",
        sample_rate=24_000,
        v_lim=(0.0, 120.0),
        fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
    )

    ads = project.prepare_audio(transform=transform)

    new_sr = 12_000
    new_name = "new_transform"
    new_instrument = Instrument(end_to_end_db=100)
    new_data = ads.data[::2]
    new_normalization = Normalization.ZSCORE

    ads.sample_rate = new_sr
    transform.sample_rate = new_sr
    transform.fft.fs = new_sr
    ads.name = new_name
    transform.name = new_name
    ads.instrument = new_instrument
    ads.data = new_data
    ads.normalization = new_normalization

    # Spectro edits
    new_v_lim = (50.0, 100.0)
    sds = project.prepare_spectro(transform=transform, audio_dataset=ads)
    sds.v_lim = new_v_lim
    for idx, sd in enumerate(sds.data):
        sd.name = str(idx)

    project.run(transform, audio_dataset=ads, spectro_dataset=sds)

    # New ads name
    assert (project.folder / "data" / "audio" / ads.name).exists()

    # New sds name
    assert (project.folder / "processed" / ads.base_name).exists()

    output_ads = AudioDataset.from_json(
        project.get_output(f"{new_name}_audio").folder / f"{new_name}_audio.json",
    )
    output_sds = SpectroDataset.from_json(
        project.get_output(new_name).folder / f"{new_name}.json",
    )

    # Only filtered data have been written
    assert len(output_ads.data) == len(new_data)
    assert len(output_sds.data) == len(new_data)

    # Analyses have the edited sr
    assert output_ads.sample_rate == new_sr
    assert output_sds.fft.fs == new_sr

    # Analyses have the edited normalization
    assert output_ads.normalization == new_normalization

    # Instrument has been edited
    assert output_ads.instrument.end_to_end_db == new_instrument.end_to_end_db

    # Spectro data have been edited
    assert output_sds.v_lim == new_v_lim
    assert all(sd.name == str(i) for i, sd in enumerate(output_sds.data))


def test_delete_output_dataset(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    project.build()

    transform_1 = Transform(
        output_type=OutputType.AUDIO | OutputType.SPECTROGRAM,
        data_duration=project.origin_dataset.duration / 2,
        name="transform_1",
        sample_rate=24_000,
        fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
    )

    transform_2 = Transform(
        output_type=OutputType.AUDIO | OutputType.SPECTROGRAM,
        data_duration=project.origin_dataset.duration / 2,
        name="transform_2",
        sample_rate=20_000,
        fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=20_000),
    )

    project.run(transform_1)
    project.run(transform_2)

    ds1 = project.get_output(transform_1.name)
    ds2 = project.get_output(transform_2.name)
    ds3 = project.get_output(f"{transform_1.name}_audio")
    ds4 = project.get_output(f"{transform_2.name}_audio")

    # Tests Project.get_output_by_transform_name
    assert project.get_output_by_transform_name("transform_1") == [ds3, ds1]
    assert project.get_output_by_transform_name("transform_2") == [ds4, ds2]

    datasets = [ds1, ds2, ds3, ds4]

    for i, ds in enumerate(datasets):
        assert ds.name in project.outputs.keys()
        assert ds.folder.exists()

        project._delete_output(str(ds.name))

        assert ds.name not in project.outputs.keys()
        assert not ds.folder.exists()

        # The JSON should be updated
        new_project = Project.from_json(project.folder / "project.json")
        assert ds.name not in new_project.outputs.keys()


@pytest.mark.parametrize(
    "transform_to_delete",
    [
        pytest.param(
            Transform(
                output_type=OutputType.AUDIO,
                data_duration=Timedelta(seconds=1),
                name="transform_to_delete",
                sample_rate=24_000,
                fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
            ),
            id="audio_only",
        ),
        pytest.param(
            Transform(
                output_type=OutputType.SPECTROGRAM,
                data_duration=Timedelta(seconds=1),
                name="transform_to_delete",
                sample_rate=24_000,
                fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
            ),
            id="spectro_only",
        ),
        pytest.param(
            Transform(
                output_type=OutputType.AUDIO | OutputType.SPECTROGRAM,
                data_duration=Timedelta(seconds=1),
                name="transform_to_delete",
                sample_rate=24_000,
                fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
            ),
            id="audio_and_spectro",
        ),
    ],
)
def test_delete_output(
    tmp_path: Path,
    audio_files: pytest.fixture,
    transform_to_delete: Transform,
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    project.build()

    # Add another transform to check that it is not affected by the deletion

    transform_to_keep = Transform(
        output_type=OutputType.AUDIO | OutputType.SPECTROGRAM | OutputType.SPECTRUM,
        data_duration=project.origin_dataset.duration / 2,
        name="transform_to_keep",
        sample_rate=24_000,
        fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
    )

    project.run(transform_to_keep)
    project.run(transform_to_delete)

    assert all(
        transform in project.transforms
        for transform in (transform_to_keep.name, transform_to_delete.name)
    )

    datasets_to_keep = project.get_output_by_transform_name(
        transform_to_keep.name,
    )
    datasets_to_delete = project.get_output_by_transform_name(
        transform_to_delete.name,
    )

    assert all(ds.folder.exists() for ds in (datasets_to_keep + datasets_to_delete))

    project.delete_transform_with_outputs(transform_to_delete.name)

    assert transform_to_keep.name in project.transforms
    assert transform_to_delete.name not in project.transforms

    deserialized_project = Project.from_json(project.folder / "project.json")

    for proj in (project, deserialized_project):
        datasets_to_keep = proj.get_output_by_transform_name(
            transform_to_keep.name,
        )
        datasets_to_delete = proj.get_output_by_transform_name(
            transform_to_delete.name,
        )

        assert all(ds.folder.exists() for ds in datasets_to_keep)
        assert not any(ds.folder.exists() for ds in datasets_to_delete)

        assert all(ds.name in proj.outputs.keys() for ds in datasets_to_keep)
        assert not any(ds.name in proj.outputs.keys() for ds in datasets_to_delete)


def test_existing_output_warning(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    project.build()

    project.run(
        Transform(
            output_type=OutputType.AUDIO,
            data_duration=project.origin_dataset.duration / 2,
            name="my_transform",
            sample_rate=24_000,
        ),
    )

    with pytest.raises(
        ValueError,
        match="A transform with the name my_transform has already been run",
    ):
        project.run(
            Transform(
                output_type=OutputType.SPECTROGRAM,
                data_duration=project.origin_dataset.duration / 2,
                name="my_transform",
                sample_rate=24_000,
                fft=ShortTimeFFT(hamming(1024), hop=1024, fs=24_000),
            ),
        )

    project.delete_transform_with_outputs("my_transform")

    project.run(
        Transform(
            output_type=OutputType.SPECTROGRAM,
            data_duration=project.origin_dataset.duration / 2,
            name="my_transform",
            sample_rate=24_000,
            fft=ShortTimeFFT(hamming(1024), hop=1024, fs=24_000),
        ),
    )


def test_rename_transform(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], None],
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    project.build()

    first_name, second_name = "fontaines", "dc"

    transform = Transform(
        output_type=OutputType.AUDIO | OutputType.SPECTROGRAM | OutputType.SPECTRUM,
        data_duration=project.origin_dataset.duration / 2,
        name=first_name,
        sample_rate=24_000,
        fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
    )

    project.run(transform)

    names = (first_name, second_name, second_name)  # Tests both renaming and same name
    for old, new in itertools.pairwise(names):
        files = {}
        for dataset in project.get_output_by_transform_name(old):
            files |= {
                file.path.name: file.path.relative_to(dataset.folder)
                for file in dataset.files
            }

        project.rename_transform_with_outputs(old, new)

        if old != new:
            assert old not in project.transforms
            assert not (project.folder / "processed" / old).exists()
            assert not (project.folder / "data" / "audio" / f"{old}_audio").exists()
            assert not project.get_output_by_transform_name(old)

        assert new in project.transforms
        assert len(project.get_output_by_transform_name(new)) == 2

        assert (project.folder / "data" / "audio" / f"{new}_audio").exists()
        assert (project.folder / "processed" / new).exists()

        assert (
            len(
                Project.from_json(
                    project.folder / "project.json",
                ).get_output_by_transform_name(
                    new,
                ),
            )
            == 2
        )

        for dataset in project.get_output_by_transform_name(new):
            for file in dataset.files:
                assert file.path.relative_to(dataset.folder) == files[file.path.name]

    # RENAME ERRORS
    with pytest.raises(ValueError, match=r"You can't rename the original dataset."):
        project.rename_transform_with_outputs(
            transform_name="original",
            new_transform_name="vampire",
        )

    with pytest.raises(ValueError, match=r"original already exists."):
        project.rename_transform_with_outputs(
            transform_name=second_name,
            new_transform_name="original",
        )

    unknown_name = "white"
    target_name = "sky"
    with pytest.raises(ValueError, match=f"Unknown output {unknown_name}."):
        project.rename_transform_with_outputs(
            transform_name=unknown_name,
            new_transform_name=target_name,
        )


def test_spectro_transform_with_existing_ads(
    tmp_path: Path,
    audio_files: tuple[list[AudioFile], None],
) -> None:
    project = Project(
        folder=tmp_path,
        strptime_format=TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED,
    )

    project.build()

    transform = Transform(
        output_type=OutputType.AUDIO,
        data_duration=project.origin_dataset.duration / 2,
        name="audio",
        sample_rate=24_000,
    )

    project.run(transform)

    transform_2 = Transform(
        OutputType.SPECTROGRAM,
        name="spectro",
        fft=ShortTimeFFT(win=hamming(1024), hop=1024, fs=24_000),
    )

    project.run(transform_2, audio_dataset=project.get_output("audio"))

    ads = project.get_output("audio")
    sds = project.get_output("spectro")

    assert type(ads) is AudioDataset
    assert type(sds) is SpectroDataset

    for ad, sd in zip(ads.data, sds.data, strict=True):
        assert ad.begin == sd.begin
        assert ad.end == sd.end
        assert sd.audio_data == ad

    with pytest.raises(ValueError, match=r"Dataset 'clafoutis' not found."):
        project.get_output("clafoutis")


def test_build_specific_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    p1, p2, p3, p4 = (
        Path(r"end.mp3"),
        Path(r"glory.mp3"),
        Path(r"slow.mp3"),
        Path(r"story.mp3"),
    )

    base_folder = [p1, p2, p3, p4]
    dest_folder = []

    project = Project(
        folder=tmp_path / "non_existing_folder",
        strptime_format="%y%m%d%H%M%S",
    )

    def mock_copyfile(file: Path, destination: Path) -> None:
        assert destination.parent == project.folder
        dest_folder.append(file)

    def mock_replace(self: Path, destination: Path) -> None:
        assert destination.parent == project.folder
        base_folder.remove(self)
        dest_folder.append(self)

    built_files = []

    def build_mock(*args: list, **kwargs: dict) -> None:
        for file in dest_folder:
            built_files.append(file)

    monkeypatch.setattr("shutil.copyfile", mock_copyfile)
    monkeypatch.setattr(Path, "replace", mock_replace)
    monkeypatch.setattr(Project, "build", build_mock)

    mkdir_calls = []

    def mkdir_mock(self: Path, *args: list, **kwargs: dict) -> None:
        mkdir_calls.append(self)

    monkeypatch.setattr(Path, "mkdir", mkdir_mock)

    assert project.folder not in mkdir_calls

    # Build from files COPY MODE
    project.build_from_files(
        (p1, p2),
    )

    assert project.folder in mkdir_calls

    assert np.array_equal(base_folder, [p1, p2, p3, p4])
    assert np.array_equal(dest_folder, [p1, p2])
    assert np.array_equal(built_files, [p1, p2])

    # Build from files MOVE MODE

    dest_folder = []
    built_files = []

    project.build_from_files(
        (p1, p2),
        move_files=True,
    )

    assert np.array_equal(base_folder, [p3, p4])
    assert np.array_equal(dest_folder, [p1, p2])
    assert np.array_equal(built_files, [p1, p2])


def test_deserialize_output_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    project = Project(
        folder=Path("grizzly"),
        strptime_format="bear",
    )

    dummy_ads = AudioDataset([])
    dummy_sds = SpectroDataset([])
    dummy_ltasds = LTASDataset([])

    project.outputs |= {
        "original": {
            "class": "AudioDataset",
            "transform": "original",
            "dataset": Path("original.json"),
        },
        "spectro": {
            "class": "SpectroDataset",
            "transform": "spectro",
            "dataset": Path("spectro.json"),
        },
        "ltas": {
            "class": "LTASDataset",
            "transform": "ltas",
            "dataset": Path("ltas.json"),
        },
    }

    json_calls = [0]

    def mock_from_json(
        mock_instance: AudioDataset | SpectroDataset | LTASDataset,
    ) -> AudioDataset | SpectroDataset | LTASDataset:
        json_calls[0] += 1
        return mock_instance

    monkeypatch.setattr(AudioDataset, "from_json", lambda _: mock_from_json(dummy_ads))
    monkeypatch.setattr(
        SpectroDataset,
        "from_json",
        lambda _: mock_from_json(dummy_sds),
    )
    monkeypatch.setattr(
        LTASDataset,
        "from_json",
        lambda _: mock_from_json(dummy_ltasds),
    )

    assert isinstance(project.outputs["original"]["dataset"], Path)

    # Getting the dataset should deserialize it
    _ = project.get_output("original")
    assert json_calls[0] == 1

    # The deserialized dataset should be stored
    assert isinstance(project.outputs["original"]["dataset"], AudioDataset)

    # Getting the dataset again should use the cached dataset
    _ = project.get_output("original")
    assert json_calls[0] == 1

    assert isinstance(project.outputs["spectro"]["dataset"], Path)

    # Getting the dataset should deserialize it
    _ = project.get_output("spectro")
    assert json_calls[0] == 2

    # The deserialized dataset should be stored
    assert isinstance(project.outputs["spectro"]["dataset"], SpectroDataset)

    # Getting the dataset again should use the cached dataset
    _ = project.get_output("spectro")
    assert json_calls[0] == 2

    assert isinstance(project.outputs["ltas"]["dataset"], Path)

    # Getting the dataset should deserialize it
    _ = project.get_output("ltas")
    assert json_calls[0] == 3

    # The deserialized dataset should be stored
    assert isinstance(project.outputs["ltas"]["dataset"], LTASDataset)

    # Getting the dataset again should use the cached dataset
    _ = project.get_output("ltas")
    assert json_calls[0] == 3
