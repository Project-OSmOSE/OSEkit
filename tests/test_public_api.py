from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from pandas import Timedelta, Timestamp

from OSmOSE.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES,
    TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
    TIMESTAMP_FORMAT_TEST_FILES,
)
from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.event import Event
from OSmOSE.public_api.dataset import Analysis, Dataset


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
    ("audio_files", "ads_name", "begin", "end", "data_duration", "sample_rate"),
    [
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            None,
            None,
            None,
            id="same_format_as_original",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            "cool",
            None,
            None,
            None,
            None,
            id="named_dataset",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            Timestamp("2024-01-01 12:00:02"),
            Timestamp("2024-01-01 12:00:04"),
            None,
            None,
            id="part_of_the_timespan",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            None,
            Timedelta(seconds=1),
            None,
            id="resize_data_with_data_duration",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            None,
            None,
            None,
            None,
            24_000,
            id="reshaping_data",
        ),
        pytest.param(
            {
                "duration": 5,
                "sample_rate": 48_000,
                "nb_files": 1,
                "date_begin": Timestamp("2024-01-01 12:00:00"),
            },
            "fun",
            Timestamp("2024-01-01 12:00:01"),
            Timestamp("2024-01-01 12:00:04"),
            Timedelta(seconds=0.5),
            24_000,
            id="full_reshape",
        ),
    ],
    indirect=["audio_files"],
)
def test_reshape(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    ads_name: str | None,
    begin: Timestamp | None,
    end: Timestamp | None,
    data_duration: Timedelta | None,
    sample_rate: float | None,
) -> None:

    dataset = Dataset(folder=tmp_path, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)
    dataset.build()
    dataset.run_analysis(
        analysis=Analysis.AUDIO,
        begin=begin,
        end=end,
        data_duration=data_duration,
        sample_rate=sample_rate,
        name=ads_name,
        subtype="DOUBLE",
    )

    expected_ads = AudioDataset.from_files(
        list(dataset.origin_dataset.files),
        begin=begin,
        end=end,
        data_duration=data_duration,
        name=ads_name,
    )
    if sample_rate is not None:
        expected_ads.sample_rate = sample_rate

    expected_ads_name = (
        ads_name
        if ads_name
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
        )
    )

    # ads folder should match the ads name
    ads_folder_name = (
        ads_name
        if ads_name
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
