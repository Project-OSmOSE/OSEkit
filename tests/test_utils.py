from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta

from osekit.core.ltas_dataset import LTASDataset
from osekit.core.spectro_dataset import SpectroDataset
from osekit.utils.audio import Butterworth, Normalization, normalize
from osekit.utils.core import (
    file_indexes_per_batch,
    get_closest_value_index,
    locked,
    nb_files_per_batch,
)
from osekit.utils.deserialization import deserialize_spectro_or_ltas_dataset
from osekit.utils.path import is_absolute, move_tree

if TYPE_CHECKING:
    from _pytest.monkeypatch import MonkeyPatch

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
    ("path", "expected"),
    [
        pytest.param(
            r"C:/tindersticks/soft/tissue",
            True,
            id="windows_absolute_path",
        ),
        pytest.param(
            r"/car/seat/headrest",
            True,
            id="unix_absolute_path",
        ),
        pytest.param(
            r"she/past/away",
            False,
            id="windows_relative_path_in_subdir",
        ),
        pytest.param(
            r"./arabia/mountain",
            False,
            id="unix_relative_path_from_current",
        ),
        pytest.param(
            r"../../part/company",
            False,
            id="unix_relative_path_from_upper_dir",
        ),
    ],
)
def test_is_absolute(path: str, expected: bool) -> None:
    assert is_absolute(path) == expected


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


def test_locked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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

    def sleep_patch(*args: None, **kwargs: None) -> None:
        msg = "Lock file present."
        raise PermissionError(msg)

    monkeypatch.setattr(time, "sleep", sleep_patch)

    # time.sleep should not be called if lock file doesn't exist
    edit_file("coolcoolcool")

    # time.sleep should be called if lock file exists
    lock_file.touch()
    with pytest.raises(PermissionError, match=r"Lock file present."):
        edit_file("")


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
    values: list[float],
    target: float,
    expected: int,
) -> None:
    assert get_closest_value_index(values=values, target=target) == expected


@pytest.mark.parametrize(
    ("normalizations", "expected"),
    [
        pytest.param(
            [Normalization.RAW],
            nullcontext(Normalization.RAW),
            id="raw_is_fine",
        ),
        pytest.param(
            [1],
            nullcontext(Normalization.RAW),
            id="int_raw_is_fine",
        ),
        pytest.param(
            [Normalization.DC_REJECT],
            nullcontext(Normalization.DC_REJECT),
            id="dc_reject_is_fine",
        ),
        pytest.param(
            [Normalization.DC_REJECT, Normalization.PEAK],
            nullcontext(Normalization.DC_REJECT | Normalization.PEAK),
            id="dc_and_peak_is_fine",
        ),
        pytest.param(
            [Normalization.DC_REJECT, Normalization.ZSCORE],
            nullcontext(Normalization.DC_REJECT | Normalization.ZSCORE),
            id="dc_and_zscore_is_fine",
        ),
        pytest.param(
            [10],
            nullcontext(Normalization.DC_REJECT | Normalization.ZSCORE),
            id="int_dc_and_zscore_is_fine",
        ),
        pytest.param(
            [Normalization.DC_REJECT, Normalization.PEAK, Normalization.ZSCORE],
            pytest.raises(ValueError),
            id="dc_reject_can_be_combined_with_only_one_other_value",
        ),
        pytest.param(
            [Normalization.PEAK, Normalization.ZSCORE],
            pytest.raises(ValueError),
            id="combination_without_dc_raises",
        ),
        pytest.param(
            [4, 8],
            pytest.raises(ValueError),
            id="int_combination_without_dc_raises",
        ),
        pytest.param(
            [12],
            pytest.raises(ValueError),
            id="int_direct_combination_without_dc_raises",
        ),
    ],
)
def test_combined_normalization(
    normalizations: list[Normalization | int],
    expected,
) -> None:
    def combine_normalizations(normalizations: list[Normalization | int]):
        normalizations = [
            Normalization(n) if type(n) is int else n for n in normalizations
        ]
        output = normalizations[0]
        for n in normalizations[1:]:
            output = output | n
        return output

    with expected as e:
        assert combine_normalizations(normalizations) == e


@pytest.mark.parametrize(
    ("values", "normalization", "expected"),
    [
        pytest.param(
            np.array([0.0, 1.0, 2.0]),
            Normalization.RAW,
            np.array([0.0, 1.0, 2.0]),
            id="raw",
        ),
        pytest.param(
            np.array([0.0, 1.0, 2.0]),
            Normalization.DC_REJECT,
            np.array([-1.0, 0.0, 1.0]),
            id="dc_reject",
        ),
        pytest.param(
            np.array([0.0, 1.0, 2.0]),
            Normalization.PEAK,
            np.array([0.0, 0.5, 1.0]),
            id="peak",
        ),
        pytest.param(
            np.array([-0.25, 0.5, 0.0]),
            Normalization.PEAK,
            np.array([-0.5, 1.0, 0.0]),
            id="peak_with_negative_values",
        ),
        pytest.param(
            np.array([-0.5, 0.25, 0.0]),
            Normalization.PEAK,
            np.array([-1.0, 0.5, 0.0]),
            id="peak_with_negative_max",
        ),
        pytest.param(
            np.array([0.0, 1.0, 2.0]),
            Normalization.ZSCORE,
            np.array([-1.224744871391589, 0.0, 1.224744871391589]),
            id="zscore",
        ),
        pytest.param(
            np.array([0.0, 2.0, 4.0]),
            Normalization.DC_REJECT | Normalization.PEAK,
            np.array([-1.0, 0.0, 1.0]),
            id="dc_reject_and_peak",
        ),
        pytest.param(
            np.array([0.0, 0.0, 0.0]),
            Normalization.PEAK,
            np.array([0.0, 0.0, 0.0]),
            id="peak_with_zero_values_shouldnt_raise",
        ),
        pytest.param(
            np.array([0.5, 0.5, 0.5]),
            Normalization.DC_REJECT | Normalization.PEAK,
            np.array([0.0, 0.0, 0.0]),
            id="peak_with_zero_values_after_dc_reject_shouldnt_raise",
        ),
        pytest.param(
            np.array([0.5, 0.5, 0.5]),
            Normalization.ZSCORE,
            np.array([0.0, 0.0, 0.0]),
            id="zscore_with_zero_std_shouldnt_raise",
        ),
    ],
)
def test_normalization(
    values: np.ndarray,
    normalization: Normalization,
    expected: np.ndarray,
) -> None:
    normalized = normalize(values=values, normalization=normalization)
    assert np.array_equal(normalized, expected)


def test_deserialize_spectro_or_ltas_dataset(monkeypatch: MonkeyPatch) -> None:
    def sds_deserialization(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "SpectroDataset"

    def ltasds_deserialization(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "LTASDataset"

    monkeypatch.setattr(SpectroDataset, "from_json", sds_deserialization)
    monkeypatch.setattr(LTASDataset, "from_json", ltasds_deserialization)

    assert deserialize_spectro_or_ltas_dataset(path=Path()) == "LTASDataset"

    def raise_key_error(*args, **kwargs) -> None:  # noqa: ANN002, ANN003
        msg = "'nb_time_bins'"
        raise KeyError(msg)

    monkeypatch.setattr(LTASDataset, "from_json", raise_key_error)

    assert deserialize_spectro_or_ltas_dataset(path=Path()) == "SpectroDataset"


def test_butter_serialization() -> None:
    N = 4
    Wn = [120.0, 200.0]
    btype = "bandpass"
    butter = Butterworth(N=N, Wn=Wn, btype=btype)
    butter2 = Butterworth.from_dict(butter.to_dict())

    assert butter.N == butter2.N
    assert butter.Wn == butter2.Wn
    assert butter.btype == butter2.btype
