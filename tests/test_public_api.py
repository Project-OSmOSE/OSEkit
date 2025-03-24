from importlib.metadata import files

import pytest
from pandas import Timestamp

from OSmOSE.core_api.event import Event
from OSmOSE.public_api.dataset import Dataset
from OSmOSE.config import DATASET_PATHS, TIMESTAMP_FORMAT_TEST_FILES
from pathlib import Path


@pytest.mark.parametrize(
    ("audio_files", "other_files", "expected_audio_events", "expected_other_files"),
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
                )
            ],
            [],
            id="one_audio_file",
        )
    ],
    indirect=["audio_files"],
)
def test_dataset_build(
    tmp_path: pytest.fixture,
    audio_files: pytest.fixture,
    other_files: list[str],
    expected_audio_events: list[Event],
    expected_other_files: list[Path],
) -> None:

    # add other files
    for file in other_files:
        (tmp_path / file).touch()

    files_before_build = list(tmp_path.rglob("*"))

    dataset = Dataset(folder=tmp_path, strptime_format=TIMESTAMP_FORMAT_TEST_FILES)

    dataset.build()

    assert not any(file.exists() for file in files_before_build)

    # Resetting the dataset should put back all original files back
    dataset.reset()
    assert sorted(str(file) for file in tmp_path.rglob("*")) == sorted(
        str(file) for file in files_before_build
    )
