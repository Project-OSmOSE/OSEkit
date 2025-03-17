from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest
from pandas import Timedelta, Timestamp

from OSmOSE.core_api.base_dataset import BaseDataset
from OSmOSE.core_api.base_file import BaseFile
from OSmOSE.core_api.event import Event


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


@pytest.mark.parametrize(
    ("files", "bound", "data_duration", "expected_data"),
    [
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            "timedelta",
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
            id="timedelta_bound_with_no_duration",
        ),
        pytest.param(
            [
                BaseFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
            ],
            "timedelta",
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
            id="timedelta_bound_with_duration",
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
            "timedelta",
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
            id="timedelta_bound_with_duration_multiple_files",
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
            id="files_bound_with_one_file",
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
            id="files_bound_with_multiple_files",
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
            id="files_bound_ignores_data_duration",
        ),
    ],
)
def test_base_dataset_file_bound(
    tmp_path: pytest.fixture,
    files: list[BaseFile],
    bound: Literal["files", "timedelta"],
    data_duration: Timedelta | None,
    expected_data: list[tuple[Event, str]],
) -> None:

    ds = BaseDataset.from_files(
        files=files,
        bound=bound,
        data_duration=data_duration,
    )

    assert all(
        d.begin == e[0].begin
        and d.end == e[0].end
        and [file.path.name for file in d.files] == e[1]
        for d, e in zip(ds.data, expected_data)
    )
