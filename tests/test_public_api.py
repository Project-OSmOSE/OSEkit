from __future__ import annotations

from copy import deepcopy

import pytest
from pandas import Timestamp

from OSmOSE.config import TIMESTAMP_FORMAT_TEST_FILES
from OSmOSE.core_api.audio_dataset import AudioDataset
from OSmOSE.core_api.event import Event
from OSmOSE.public_api.dataset import Dataset


@pytest.mark.parametrize(
    ("audio_files", "other_files", "expected_audio_events"),
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
    ],
    indirect=["audio_files"],
)
def test_dataset_build(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    other_files: list[str],
    expected_audio_events: list[Event],
) -> None:

    # add other files
    for file in other_files:
        (tmp_path / file).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / file).touch()

    files_before_build = list(tmp_path.rglob("*"))
    original_dataset = AudioDataset.from_folder(
        tmp_path,
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
        bound="files",
    )

    dataset = Dataset(folder=tmp_path, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)

    dataset.build()

    assert not any(file.exists() for file in files_before_build)

    # Expected moved original dataset
    moved_original_dataset = deepcopy(original_dataset)
    for file in moved_original_dataset.files:
        file.path = file.path.parent / "data" / "audio" / "original" / file.path.name

    # The original dataset should be moved to the correct folder
    assert moved_original_dataset == AudioDataset.from_folder(
        folder=tmp_path / "data" / "audio" / "original",
        strptime_format=TIMESTAMP_FORMAT_TEST_FILES,
        bound="files",
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
