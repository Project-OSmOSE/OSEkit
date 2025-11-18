from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pandas import Timedelta, Timestamp

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.core_api.base_data import BaseData
from osekit.core_api.base_dataset import BaseDataset
from osekit.core_api.base_file import BaseFile
from osekit.core_api.event import Event


@pytest.mark.parametrize(
    ("base_files", "begin", "end", "duration", "expected_data_events"),
    [
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            None,
            None,
            None,
            [
                Event(
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            id="one_file_one_data",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            Timestamp("2016-02-05 00:00:00"),
            Timestamp("2016-02-05 01:00:00"),
            None,
            [
                Event(
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            id="one_file_one_data_explicit_begin_end",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            None,
            None,
            Timedelta(hours=1),
            [
                Event(
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            id="one_file_one_data_explicit_duration",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            Timestamp("2016-02-04 23:00:00"),
            Timestamp("2016-02-05 02:00:00"),
            None,
            [
                Event(
                    begin=Timestamp("2016-02-04 23:00:00"),
                    end=Timestamp("2016-02-05 02:00:00"),
                ),
            ],
            id="begin_end_exceed_files_boundaries",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            None,
            None,
            Timedelta(hours=2),
            [
                Event(
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 02:00:00"),
                ),
            ],
            id="duration_exceeds_files_duration",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            Timestamp("2016-02-05 00:30:00"),
            Timestamp("2016-02-05 00:31:00"),
            None,
            [
                Event(
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:31:00"),
                ),
            ],
            id="one_file_one_data_explicit_begin_end_parts",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            Timestamp("2016-02-05 00:30:00"),
            Timestamp("2016-02-05 00:31:00"),
            Timedelta(seconds=20),
            [
                Event(
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:30:20"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:30:20"),
                    end=Timestamp("2016-02-05 00:30:40"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:30:40"),
                    end=Timestamp("2016-02-05 00:31:00"),
                ),
            ],
            id="one_file_one_data_explicit_begin_end_duration_parts",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:50:00"),
                ),
            ],
            None,
            None,
            Timedelta(minutes=20),
            [
                Event(
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:20:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:20:00"),
                    end=Timestamp("2016-02-05 00:40:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:40:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            id="one_file_several_data_with_modulo",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:10:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:10:00"),
                    end=Timestamp("2016-02-05 00:20:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:20:00"),
                    end=Timestamp("2016-02-05 00:30:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:40:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:40:00"),
                    end=Timestamp("2016-02-05 00:50:00"),
                ),
            ],
            None,
            None,
            Timedelta(minutes=20),
            [
                Event(
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:20:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:20:00"),
                    end=Timestamp("2016-02-05 00:40:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:40:00"),
                    end=Timestamp("2016-02-05 01:00:00"),
                ),
            ],
            id="several_files_several_data_with_modulo",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:10:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:10:00"),
                    end=Timestamp("2016-02-05 00:20:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:20:00"),
                    end=Timestamp("2016-02-05 00:30:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:40:00"),
                ),
                BaseFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:40:00"),
                    end=Timestamp("2016-02-05 00:50:00"),
                ),
            ],
            Timestamp("2016-02-04 23:50:00"),
            Timestamp("2016-02-05 01:00:00"),
            Timedelta(minutes=20),
            [
                Event(
                    begin=Timestamp("2016-02-04 23:50:00"),
                    end=Timestamp("2016-02-05 00:10:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:10:00"),
                    end=Timestamp("2016-02-05 00:30:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:50:00"),
                ),
                Event(
                    begin=Timestamp("2016-02-05 00:50:00"),
                    end=Timestamp("2016-02-05 01:10:00"),
                ),
            ],
            id="several_files_several_data_with_begin_end_and_modulo_duration",
        ),
    ],
)
def test_base_dataset_from_files(
    base_files: list[BaseFile],
    begin: Timestamp | None,
    end: Timestamp | None,
    duration: Timedelta | None,
    expected_data_events: list[Event],
) -> None:
    ads = BaseDataset.from_files(
        base_files,
        begin=begin,
        end=end,
        data_duration=duration,
    )
    assert expected_data_events == [Event(d.begin, d.end) for d in ads.data]


@pytest.mark.parametrize(
    ("overlap", "mode"),
    [
        pytest.param(
            -1.0,
            "timedelta_files",
            id="negative_overlap_files",
        ),
        pytest.param(
            1.0,
            "timedelta_files",
            id="one_overlap_files",
        ),
        pytest.param(
            10.0,
            "timedelta_files",
            id="greater_than_one_overlap_files",
        ),
        pytest.param(
            -1.0,
            "timedelta_total",
            id="negative_overlap_total",
        ),
        pytest.param(
            1.0,
            "timedelta_total",
            id="one_overlap_total",
        ),
        pytest.param(
            10.0,
            "timedelta_total",
            id="greater_than_one_overlap_total",
        ),
    ],
)
def test_base_dataset_from_files_overlap_errors(overlap: float, mode: str) -> None:
    with pytest.raises(
        ValueError,
        match=rf"Overlap \({overlap}\) must be between 0 and 1.",
    ) as e:
        assert (
            BaseDataset.from_files(
                [
                    BaseFile(
                        path=Path("foo"),
                        begin=Timestamp("2016-02-05 00:00:00"),
                        end=Timestamp("2016-02-05 00:10:00"),
                    ),
                ],
                data_duration=Timedelta(seconds=1),
                mode=mode,
                overlap=overlap,
            )
            == e
        )


@pytest.mark.parametrize(
    "destination_folder",
    [
        pytest.param(
            "cool",
            id="moving_to_new_folder",
        ),
        pytest.param(
            "",
            id="moving_to_same_folder",
        ),
    ],
)
def test_move_file(
    tmp_path: Path,
    destination_folder: str,
) -> None:
    filename = "cool.txt"
    (tmp_path / filename).touch(mode=0o666, exist_ok=True)
    bf = BaseFile(
        tmp_path / filename,
        begin=Timestamp("2022-04-22 12:12:12"),
        end=Timestamp("2022-04-22 12:13:12"),
    )

    bf.move(tmp_path / destination_folder)

    assert (tmp_path / destination_folder / filename).exists()

    if destination_folder:
        assert not (tmp_path / filename).exists()


def test_dataset_move(
    tmp_path: Path,
    base_dataset: BaseDataset,
) -> None:
    origin_files = [Path(str(file.path)) for file in base_dataset.files]

    # The starting folder of the dataset is the folder where the files are located
    assert base_dataset.folder == tmp_path

    destination = tmp_path / "destination"
    base_dataset.folder = destination

    # Setting the folder shouldn't move the files
    assert all(file.path.parent == tmp_path for file in base_dataset.files)
    assert all(file.exists for file in origin_files)

    # Folder should be changed when dataset is moved
    new_destination = tmp_path / "new_destination"
    base_dataset.move_files(new_destination)

    assert new_destination.exists()
    assert new_destination.is_dir()
    assert base_dataset.folder == new_destination
    assert all((new_destination / file.name).exists() for file in origin_files)
    assert not any(file.exists() for file in origin_files)


@pytest.mark.parametrize(
    ("files", "mode", "data_duration", "expected_data"),
    [
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            "timedelta_total",
            None,
            [
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12"),
                        end=Timestamp("2022-04-22 12:12:13"),
                    ),
                    ["cool"],
                ),
            ],
            id="timedelta_mode_with_no_duration",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            "timedelta_total",
            Timedelta(seconds=0.5),
            [
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12"),
                        end=Timestamp("2022-04-22 12:12:12.5"),
                    ),
                    ["cool"],
                ),
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12.5"),
                        end=Timestamp("2022-04-22 12:12:13"),
                    ),
                    ["cool"],
                ),
            ],
            id="timedelta_mode_with_duration",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                BaseFile(
                    path=Path("fun"),
                    begin=Timestamp("2022-04-22 12:12:13"),
                    end=Timestamp("2022-04-22 12:12:14"),
                ),
            ],
            "timedelta_total",
            Timedelta(seconds=0.5),
            [
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12"),
                        end=Timestamp("2022-04-22 12:12:12.5"),
                    ),
                    ["cool"],
                ),
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12.5"),
                        end=Timestamp("2022-04-22 12:12:13"),
                    ),
                    ["cool"],
                ),
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:13"),
                        end=Timestamp("2022-04-22 12:12:13.5"),
                    ),
                    ["fun"],
                ),
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:13.5"),
                        end=Timestamp("2022-04-22 12:12:14"),
                    ),
                    ["fun"],
                ),
            ],
            id="timedelta_mode_with_duration_multiple_files",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            "files",
            None,
            [
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12"),
                        end=Timestamp("2022-04-22 12:12:13"),
                    ),
                    ["cool"],
                ),
            ],
            id="files_mode_with_one_file",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                BaseFile(
                    path=Path("fun"),
                    begin=Timestamp("2022-04-22 12:12:13"),
                    end=Timestamp("2022-04-22 12:12:14"),
                ),
            ],
            "files",
            None,
            [
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12"),
                        end=Timestamp("2022-04-22 12:12:13"),
                    ),
                    ["cool"],
                ),
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:13"),
                        end=Timestamp("2022-04-22 12:12:14"),
                    ),
                    ["fun"],
                ),
            ],
            id="files_mode_with_multiple_files",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                BaseFile(
                    path=Path("fun"),
                    begin=Timestamp("2022-04-22 12:12:13"),
                    end=Timestamp("2022-04-22 12:12:14"),
                ),
            ],
            "files",
            Timedelta(seconds=0.1),
            [
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:12"),
                        end=Timestamp("2022-04-22 12:12:13"),
                    ),
                    ["cool"],
                ),
                (
                    Event(
                        begin=Timestamp("2022-04-22 12:12:13"),
                        end=Timestamp("2022-04-22 12:12:14"),
                    ),
                    ["fun"],
                ),
            ],
            id="files_mode_ignores_data_duration",
        ),
    ],
)
def test_base_dataset_file_mode(
    tmp_path: pytest.fixture,
    files: list[BaseFile],
    mode: Literal["files", "timedelta_total"],
    data_duration: Timedelta | None,
    expected_data: list[tuple[Event, str]],
) -> None:
    ds = BaseDataset.from_files(
        files=files,
        mode=mode,
        data_duration=data_duration,
    )

    assert all(
        d.begin == e[0].begin
        and d.end == e[0].end
        and [file.path.name for file in d.files] == e[1]
        for d, e in zip(ds.data, expected_data, strict=False)
    )


@pytest.mark.parametrize(
    ("files", "begin", "end", "expected_data"),
    [
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            None,
            None,
            Event(
                begin=Timestamp("2022-04-22 12:12:12"),
                end=Timestamp("2022-04-22 12:12:13"),
            ),
            id="no_boundary_change",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            Timestamp("2022-04-22 12:12:12.5"),
            None,
            Event(
                begin=Timestamp("2022-04-22 12:12:12.5"),
                end=Timestamp("2022-04-22 12:12:13"),
            ),
            id="begin_after_start",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            Timestamp("2022-04-22 12:12:11.5"),
            None,
            Event(
                begin=Timestamp("2022-04-22 12:12:12"),
                end=Timestamp("2022-04-22 12:12:13"),
            ),
            id="begin_before_start_has_no_effect",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            None,
            Timestamp("2022-04-22 12:12:12.5"),
            Event(
                begin=Timestamp("2022-04-22 12:12:12"),
                end=Timestamp("2022-04-22 12:12:12.5"),
            ),
            id="end_before_stop",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            None,
            Timestamp("2022-04-22 12:12:14"),
            Event(
                begin=Timestamp("2022-04-22 12:12:12"),
                end=Timestamp("2022-04-22 12:12:13"),
            ),
            id="end_after_stop_has_no_effect",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                BaseFile(
                    path=Path("fun"),
                    begin=Timestamp("2022-04-22 12:12:14"),
                    end=Timestamp("2022-04-22 12:12:15"),
                ),
            ],
            Timestamp("2022-04-22 12:12:12.5"),
            Timestamp("2022-04-22 12:12:14.5"),
            Event(
                begin=Timestamp("2022-04-22 12:12:12.5"),
                end=Timestamp("2022-04-22 12:12:14.5"),
            ),
            id="valid_change_within_multiple_files",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                BaseFile(
                    path=Path("fun"),
                    begin=Timestamp("2022-04-22 12:12:14"),
                    end=Timestamp("2022-04-22 12:12:15"),
                ),
            ],
            Timestamp("2022-04-22 12:12:13.1"),
            None,
            Event(
                begin=Timestamp("2022-04-22 12:12:13.1"),
                end=Timestamp("2022-04-22 12:12:15"),
            ),
            id="valid_change_excluding_one_files",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                BaseFile(
                    path=Path("fun"),
                    begin=Timestamp("2022-04-22 12:12:14"),
                    end=Timestamp("2022-04-22 12:12:15"),
                ),
            ],
            Timestamp("2022-04-22 12:12:13.1"),
            Timestamp("2022-04-22 12:12:13.9"),
            Event(
                begin=Timestamp("2022-04-22 12:12:13.1"),
                end=Timestamp("2022-04-22 12:12:13.9"),
            ),
            id="valid_change_excluding_all_files",
        ),
    ],
)
def test_base_data_boundaries(
    monkeypatch: pytest.fixture,
    files: list[BaseFile],
    begin: Timestamp,
    end: Timestamp,
    expected_data: Event,
) -> None:
    data = BaseData.from_files(files=files)
    if begin:
        data.begin = begin
    if end:
        data.end = end
    assert data.begin == expected_data.begin
    assert data.end == expected_data.end

    def mocked_get_value(self: BaseData) -> None:
        for item in data.items:
            if item.is_empty:
                continue
            assert item.file.begin <= item.begin
            assert item.file.end >= item.end

    monkeypatch.setattr(BaseData, "get_value", mocked_get_value)

    data.get_value()


@pytest.mark.parametrize(
    ("data1", "data2", "expected"),
    [
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            True,
            id="same_one_full_file",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=Timestamp("2015-08-28 12:12:12"),
                end=Timestamp("2015-08-28 12:13:12"),
            ),
            True,
            id="same_one_full_file_explicit_timestamps",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=Timestamp("2015-08-28 12:12:13"),
                end=None,
            ),
            False,
            id="different_begin",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=Timestamp("2015-08-28 12:13:10"),
            ),
            False,
            id="different_end",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=Timestamp("2015-08-28 12:12:13"),
                end=Timestamp("2015-08-28 12:13:10"),
            ),
            False,
            id="different_begin_and_end",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            False,
            id="different_file",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    BaseFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:14"),
                        end=Timestamp("2015-08-28 12:13:15"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    BaseFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:14"),
                        end=Timestamp("2015-08-28 12:13:15"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            True,
            id="same_two_files",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    BaseFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:14"),
                        end=Timestamp("2015-08-28 12:13:15"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    BaseFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:14"),
                        end=Timestamp("2015-08-28 12:13:15"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            False,
            id="different_one_out_of_two_files",
        ),
    ],
)
def test_base_data_equality(data1: BaseData, data2: BaseData, expected: bool) -> None:
    assert (data1 == data2) == expected


@pytest.mark.parametrize(
    ("data", "name", "expected"),
    [
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
            ),
            None,
            Timestamp("2015-08-28 12:12:12").strftime(
                TIMESTAMP_FORMATS_EXPORTED_FILES[0],
            ),
            id="default_to_data_begin",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:13:12"),
                        end=Timestamp("2015-08-28 12:14:12"),
                    ),
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=Timestamp("2015-08-28 12:12:30"),
                end=Timestamp("2015-08-28 12:13:30"),
            ),
            None,
            Timestamp("2015-08-28 12:12:30").strftime(
                TIMESTAMP_FORMATS_EXPORTED_FILES[0],
            ),
            id="default_to_data_begin_with_unordered_files",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
            ),
            "cool_raoul",
            "cool_raoul",
            id="given_name",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                name="uncool_raoul",
            ),
            "cool_raoul",
            "cool_raoul",
            id="given_name_over_existing_name",
        ),
        pytest.param(
            BaseData.from_files(
                [
                    BaseFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                name="uncool_raoul",
            ),
            None,
            Timestamp("2015-08-28 12:12:12").strftime(
                TIMESTAMP_FORMATS_EXPORTED_FILES[0],
            ),
            id="none_resets_to_default",
        ),
    ],
)
def test_data_name(data: BaseData, name: str | None, expected: str) -> None:
    data.name = name
    assert data.name == expected
    assert str(data) == expected


@pytest.mark.parametrize(
    ("files", "begin", "end", "data_duration", "mode", "overlap", "expected"),
    [
        pytest.param(
            [
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:13:12"),
            Timedelta(seconds=30),
            "timedelta_total",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:12:42"),
                        ),
                        "cherry",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:42"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="total_only_one_file",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:12"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_total",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="total_two_continuous_files_without_file_overlap",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:12"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(seconds=45),
            "timedelta_total",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:12:57"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:57"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:42"),
                        ),
                        "cherry",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:42"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:14:12"),
                            end=Timestamp("2015-08-28 12:14:27"),
                        ),
                        None,
                    ),
                ],
            ],
            id="total_two_continuous_files_with_file_overlap",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:12"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_total",
            0.25,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:57"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:57"),
                        ),
                        "cherry",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:42"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:14:12"),
                            end=Timestamp("2015-08-28 12:14:42"),
                        ),
                        None,
                    ),
                ],
            ],
            id="total_two_continuous_files_with_data_overlap",
        ),
        pytest.param(
            [
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:13:12"),
            Timedelta(seconds=30),
            "timedelta_file",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:12:42"),
                        ),
                        "cherry",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:42"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="file_only_one_file",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:12"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_file",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="file_two_continuous_files_without_file_overlap",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:12"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(seconds=45),
            "timedelta_file",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:12:57"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:57"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:42"),
                        ),
                        "cherry",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:42"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:14:12"),
                            end=Timestamp("2015-08-28 12:14:27"),
                        ),
                        None,
                    ),
                ],
            ],
            id="file_two_continuous_files_with_file_overlap",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:22"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:32"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_total",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:22"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:22"),
                            end=Timestamp("2015-08-28 12:13:32"),
                        ),
                        None,
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:32"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="total_two_separate_files_without_empty_start",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:22"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:32"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_file",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:22"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:22"),
                            end=Timestamp("2015-08-28 12:13:32"),
                        ),
                        None,
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:32"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="file_two_separate_files_without_empty_start",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:22"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_total",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:02"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:02"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        None,
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:22"),
                        ),
                        None,
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:22"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="total_two_separate_files_with_empty_start",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:22"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_file",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:02"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:02"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        None,
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:22"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:14:12"),
                            end=Timestamp("2015-08-28 12:14:22"),
                        ),
                        None,
                    ),
                ],
            ],
            id="file_two_separate_files_with_empty_start",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:00"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:22"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_total",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:12"),
                            end=Timestamp("2015-08-28 12:13:02"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:02"),
                            end=Timestamp("2015-08-28 12:13:12"),
                        ),
                        None,
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:12"),
                            end=Timestamp("2015-08-28 12:13:22"),
                        ),
                        None,
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:22"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                ],
            ],
            id="total_two_separate_files_begin_mid_file",
        ),
        pytest.param(
            [
                BaseFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:00"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                BaseFile(
                    "cherry",
                    begin=Timestamp("2015-08-28 12:13:22"),
                    end=Timestamp("2015-08-28 12:14:12"),
                ),
            ],
            Timestamp("2015-08-28 12:12:12"),
            Timestamp("2015-08-28 12:14:12"),
            Timedelta(minutes=1),
            "timedelta_file",
            0.0,
            [
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:12:00"),
                            end=Timestamp("2015-08-28 12:13:00"),
                        ),
                        "depression",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:00"),
                            end=Timestamp("2015-08-28 12:13:02"),
                        ),
                        "depression",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:02"),
                            end=Timestamp("2015-08-28 12:13:22"),
                        ),
                        None,
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:13:22"),
                            end=Timestamp("2015-08-28 12:14:00"),
                        ),
                        "cherry",
                    ),
                ],
                [
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:14:00"),
                            end=Timestamp("2015-08-28 12:14:12"),
                        ),
                        "cherry",
                    ),
                    (
                        Event(
                            begin=Timestamp("2015-08-28 12:14:12"),
                            end=Timestamp("2015-08-28 12:15:00"),
                        ),
                        None,
                    ),
                ],
            ],
            id="file_two_separate_files_begin_mid_file",
        ),
    ],
)
def test_get_base_data_from_files(
    files: list[BaseFile],
    begin: Timestamp,
    end: Timestamp,
    data_duration: Timedelta,
    mode: Literal["timedelta_total", "timedelta_file"],
    overlap: float,
    expected: list[list[tuple[Event, str | None]]],
) -> None:
    data = BaseDataset.from_files(
        files=files,
        begin=begin,
        end=end,
        mode=mode,
        overlap=overlap,
        data_duration=data_duration,
    ).data

    for d, e in zip(data, expected, strict=True):
        for item, expected_tuple in zip(d.items, e, strict=True):
            expected_item, expected_filename = expected_tuple
            assert item.begin == expected_item.begin
            assert item.end == expected_item.end
            assert (
                expected_filename is None
                if item.is_empty
                else item.file.path.name == expected_filename
            )
