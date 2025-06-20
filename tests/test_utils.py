from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from OSmOSE.utils.core_utils import (
    file_indexes_per_batch,
    get_closest_value_index,
    locked,
    nb_files_per_batch,
)
from OSmOSE.utils.formatting_utils import aplose2raven
from OSmOSE.utils.path_utils import move_tree


@pytest.fixture
def aplose_dataframe() -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "dataset": ["dataset_test", "dataset_test", "dataset_test"],
            "filename": ["file1.wav", "file2.wav", "file3.wav"],
            "start_time": [0, 0, 5.9],
            "end_time": [60, 60, 8.1],
            "start_frequency": [0, 0, 18500.0],
            "end_frequency": [96000, 96000, 53000.0],
            "annotation": ["boat", "boat", "boat"],
            "annotator": ["bbjuni", "bbjuni", "bbjuni"],
            "start_datetime": [
                pd.Timestamp("2020-05-29T11:30:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:31:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:31:05.900+00:00"),
            ],
            "end_datetime": [
                pd.Timestamp("2020-05-29T11:31:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:32:00.000+00:00"),
                pd.Timestamp("2020-05-29T11:32:08.100+00:00"),
            ],
            "is_box": [0, 0, 1],
        },
    )

    return data.reset_index(drop=True)


@pytest.fixture
def raven_timestamps() -> list:
    return list(
        pd.date_range(
            start="2020-05-29T11:30:00.000+00:00",
            end="2020-05-29T11:35:00.000+00:00",
            freq="1min",
        ),
    )


@pytest.fixture
def raven_durations(raven_timestamps: pytest.fixture) -> list:
    return [60] * len(raven_timestamps)


@pytest.mark.unit
def test_aplose2raven(
    aplose_dataframe: pytest.fixture,
    raven_timestamps: pytest.fixture,
    raven_durations: pytest.fixture,
) -> None:
    raven_dataframe = aplose2raven(
        aplose_result=aplose_dataframe,
        audio_datetimes=raven_timestamps,
        audio_durations=raven_durations,
    )

    expected_raven_dataframe = pd.DataFrame(
        {
            "Selection": [1, 2, 3],
            "View": [1, 1, 1],
            "Channel": [1, 1, 1],
            "Begin Time (s)": [0.0, 60.0, 65.9],
            "End Time (s)": [60.0, 120.0, 128.1],
            "Low Freq (Hz)": [0.0, 0.0, 18500.0],
            "High Freq (Hz)": [96000.0, 96000.0, 53000.0],
        },
    )

    assert expected_raven_dataframe.equals(raven_dataframe)


@pytest.mark.parametrize(
    ("files", "destination", "excluded_files"),
    [
        pytest.param(
            {"cool"},
            "output",
            set(),
            id="one_moved_file",
        ),
        pytest.param(
            {"cool", "fun"},
            "output",
            {"cool"},
            id="both_included_and_excluded",
        ),
        pytest.param(
            {"cool"},
            "output",
            {"cool"},
            id="all_excluded_file",
        ),
        pytest.param(
            {"cool/fun"},
            "output",
            set(),
            id="one_recursive_moving",
        ),
        pytest.param(
            {"cool/fun", "megacool/top"},
            "output",
            {"cool"},
            id="recursive_moving_with_exclusions",
        ),
        pytest.param(
            {"cool"},
            "output/fun",
            set(),
            id="moving_to_subfolder",
        ),
    ],
)
def test_move_tree(
    tmp_path: pytest.fixture,
    files: set[str],
    destination: Path,
    excluded_files: set[str],
) -> None:
    for f in files:
        (tmp_path / f).parent.mkdir(exist_ok=True, parents=True)
        (tmp_path / f).touch()

    unmoved_files = {
        file
        for file in files
        if any(
            Path(unmoved) in Path(file).parents or unmoved == file
            for unmoved in excluded_files
        )
    }

    destination = tmp_path / destination

    move_tree(tmp_path, destination, {tmp_path / file for file in excluded_files})

    assert all(not (destination / file).exists() for file in unmoved_files)
    assert all((tmp_path / file).exists() for file in unmoved_files)

    assert all((destination / file).exists() for file in files - unmoved_files)
    assert all(not (tmp_path / file).exists() for file in files - unmoved_files)

    if not files - unmoved_files:
        assert not destination.exists()


@pytest.mark.parametrize(
    ("total_nb_files", "nb_batches", "expected"),
    [
        pytest.param(
            10,
            1,
            [10],
            id="only_one_batch",
        ),
        pytest.param(
            10,
            5,
            [2, 2, 2, 2, 2],
            id="no_remainder",
        ),
        pytest.param(
            11,
            5,
            [3, 2, 2, 2, 2],
            id="first_batch_has_remainder",
        ),
        pytest.param(
            13,
            5,
            [3, 3, 3, 2, 2],
            id="remainder_is_fairly_distributed",
        ),
        pytest.param(
            3,
            5,
            [1, 1, 1, 0, 0],
            id="more_jobs_than_files",
        ),
        pytest.param(
            0,
            5,
            [0, 0, 0, 0, 0],
            id="no_file",
        ),
        pytest.param(
            5,
            0,
            [],
            id="no_job",
        ),
    ],
)
def test_nb_files_per_batch(
    total_nb_files: int,
    nb_batches: int,
    expected: list[int],
) -> None:
    assert (
        nb_files_per_batch(total_nb_files=total_nb_files, nb_batches=nb_batches)
        == expected
    )


@pytest.mark.parametrize(
    ("total_nb_files", "nb_batches", "expected"),
    [
        pytest.param(
            10,
            1,
            [(0, 10)],
            id="only_one_batch",
        ),
        pytest.param(
            10,
            5,
            [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)],
            id="no_remainder",
        ),
        pytest.param(
            11,
            5,
            [(0, 3), (3, 5), (5, 7), (7, 9), (9, 11)],
            id="first_batch_has_remainder",
        ),
        pytest.param(
            13,
            5,
            [(0, 3), (3, 6), (6, 9), (9, 11), (11, 13)],
            id="remainder_is_fairly_distributed",
        ),
        pytest.param(
            3,
            5,
            [(0, 1), (1, 2), (2, 3)],
            id="more_jobs_than_files_should_cut_unused_batches",
        ),
        pytest.param(
            0,
            5,
            [],
            id="no_file",
        ),
        pytest.param(
            5,
            0,
            [],
            id="no_job",
        ),
    ],
)
def test_file_indexes_per_batch(
    total_nb_files: int,
    nb_batches: int,
    expected: list[tuple[int, int]],
) -> None:
    assert (
        file_indexes_per_batch(total_nb_files=total_nb_files, nb_batches=nb_batches)
        == expected
    )


def test_locked(tmp_path: pytest.fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    file = tmp_path / "file.txt"
    lock_file = tmp_path / "lock.lock"

    file.touch()

    @locked(lock_file=lock_file)
    def edit_file(line_to_add: str) -> None:
        # locked decorator should create the lock file
        assert lock_file.exists()

        with file.open("a") as f:
            f.write(line_to_add)

    assert not lock_file.exists()

    edit_file("yoyoyo")

    # Lock file should be released
    assert not lock_file.exists()

    # Decorated function should have been called
    with file.open("r") as f:
        assert "yoyoyo" in f.read()

    def sleep_patch(*args: any, **kwargs: any) -> None:
        msg = "Lock file present."
        raise PermissionError(msg)

    monkeypatch.setattr(time, "sleep", sleep_patch)

    # time.sleep should not be called if lock file doesn't exist
    edit_file("coolcoolcool")

    # time.sleep should be called if lock file exists
    lock_file.touch()
    with pytest.raises(PermissionError, match="Lock file present.") as e:
        assert edit_file("") == e


@pytest.mark.parametrize(
    ("values", "target", "expected"),
    [
        pytest.param(
            list(range(10)),
            5,
            5,
            id="target_is_in_values",
        ),
        pytest.param(
            list(range(10)),
            5.2,
            5,
            id="target_is_closer_to_floor",
        ),
        pytest.param(
            list(range(10)),
            5.6,
            6,
            id="target_is_closer_to_ceiling",
        ),
        pytest.param(
            list(range(10, 20)),
            5,
            0,
            id="target_is_smaller_than_first_item",
        ),
        pytest.param(
            list(range(10, 20)),
            30,
            9,
            id="target_is_greater_than_last_item",
        ),
    ],
)
def test_get_closest_value_index(
    values: list[float], target: float, expected: int
) -> None:
    assert get_closest_value_index(values=values, target=target) == expected
