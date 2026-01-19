from __future__ import annotations

import typing
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd
import pytest
from pandas import Timedelta, Timestamp

from osekit.config import TIMESTAMP_FORMATS_EXPORTED_FILES
from osekit.core_api.base_data import BaseData, TFile
from osekit.core_api.base_dataset import BaseDataset, TData
from osekit.core_api.base_file import BaseFile
from osekit.core_api.base_item import BaseItem
from osekit.core_api.event import Event


class DummyFile(BaseFile):
    supported_extensions: typing.ClassVar = [""]

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray: ...


class DummyItem(BaseItem[DummyFile]): ...


class DummyData(BaseData[DummyItem, DummyFile]):
    item_cls = DummyItem

    def write(self, folder: Path, link: bool = False) -> None: ...

    def link(self, folder: Path) -> None: ...

    def _make_split_data(
        self,
        files: list[DummyFile],
        begin: Timestamp,
        end: Timestamp,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        return DummyData.from_files(files=files, begin=begin, end=end, **kwargs)

    @classmethod
    def _make_file(cls, path: Path, begin: Timestamp) -> DummyFile:
        return DummyFile(path=path, begin=begin)

    @classmethod
    def _make_item(
        cls,
        file: TFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> DummyItem:
        return DummyItem(file=file, begin=begin, end=end)

    @classmethod
    def _from_base_dict(
        cls,
        dictionary: dict,
        files: list[TFile],
        begin: Timestamp,
        end: Timestamp,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        return cls.from_files(
            files=files,
            begin=begin,
            end=end,
        )

    @classmethod
    def from_files(
        cls,
        files: list[DummyFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        return super().from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
            **kwargs,
        )


class DummyDataset(BaseDataset[DummyData, DummyFile]):
    @classmethod
    def _data_from_dict(cls, dictionary: dict) -> list[TData]:
        return [DummyData.from_dict(data) for data in dictionary.values()]

    @classmethod
    def _data_from_files(
        cls,
        files: list[DummyFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,
    ) -> TData:
        return DummyData.from_files(
            files=files,
            begin=begin,
            end=end,
            name=name,
        )

    file_cls = DummyFile


@pytest.fixture
def dummy_dataset(tmp_path: Path) -> DummyDataset:
    files = [tmp_path / f"file_{i}.txt" for i in range(5)]
    for file in files:
        file.touch()
    timestamps = pd.date_range(
        start=pd.Timestamp("2000-01-01 00:00:00"),
        freq="1s",
        periods=5,
    )

    dfs = [
        DummyFile(path=file, begin=timestamp, end=timestamp + pd.Timedelta(seconds=1))
        for file, timestamp in zip(files, timestamps, strict=False)
    ]
    return DummyDataset.from_files(files=dfs, mode="files")


def test_base_file_with_no_begin_error() -> None:
    with pytest.raises(
        ValueError,
        match=r"Either begin or strptime_format must be specified",
    ) as e:
        assert DummyFile(path=r"foo") == e


@pytest.mark.parametrize(
    ("f1", "f2", "expected"),
    [
        pytest.param(
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:00")),
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:00")),
            True,
            id="equal_files",
        ),
        pytest.param(
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:00")),
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:01")),
            False,
            id="different_begin_means_unequal",
        ),
        pytest.param(
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:00")),
            DummyFile(path=Path("bar"), begin=Timestamp("2016-02-05 00:00:00")),
            False,
            id="different_path_means_unequal",
        ),
        pytest.param(
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:00")),
            DummyFile(path=Path("bar"), begin=Timestamp("2016-02-05 00:00:01")),
            False,
            id="different_path_and_begin_means_unequal",
        ),
        pytest.param(
            DummyFile(path=Path("foo"), begin=Timestamp("2016-02-05 00:00:00")),
            Path(r"foo"),
            False,
            id="BaseFile_unequals_other_type",
        ),
    ],
)
def test_base_file_equality(f1: DummyFile, f2: DummyFile, expected: bool) -> None:
    assert (f1 == f2) == expected


@pytest.mark.parametrize(
    ("base_files", "begin", "end", "duration", "expected_data_events"),
    [
        pytest.param(
            [
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:10:00"),
                ),
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:10:00"),
                    end=Timestamp("2016-02-05 00:20:00"),
                ),
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:20:00"),
                    end=Timestamp("2016-02-05 00:30:00"),
                ),
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:40:00"),
                ),
                DummyFile(
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
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:00:00"),
                    end=Timestamp("2016-02-05 00:10:00"),
                ),
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:10:00"),
                    end=Timestamp("2016-02-05 00:20:00"),
                ),
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:20:00"),
                    end=Timestamp("2016-02-05 00:30:00"),
                ),
                DummyFile(
                    path=Path("foo"),
                    begin=Timestamp("2016-02-05 00:30:00"),
                    end=Timestamp("2016-02-05 00:40:00"),
                ),
                DummyFile(
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
    base_files: list[DummyFile],
    begin: Timestamp | None,
    end: Timestamp | None,
    duration: Timedelta | None,
    expected_data_events: list[Event],
) -> None:
    ads = DummyDataset.from_files(
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
            DummyDataset.from_files(
                [
                    DummyFile(
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
    (
        "strptime_format",
        "begin",
        "end",
        "timezone",
        "mode",
        "overlap",
        "data_duration",
        "first_file_begin",
        "name",
        "files",
        "expected_data_events",
    ),
    [
        pytest.param(
            "%y%m%d%H%M%S",
            None,
            None,
            None,
            "files",
            0.0,
            None,
            None,
            None,
            [Path(r"231201000000")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"231201000000")],
                ),
            ],
            id="one_file_default",
        ),
        pytest.param(
            None,
            None,
            None,
            None,
            "files",
            0.0,
            None,
            Timestamp("2023-12-01 00:00:00"),
            None,
            [Path(r"bbjuni")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"bbjuni")],
                ),
            ],
            id="one_file_no_strptime",
        ),
        pytest.param(
            None,
            None,
            None,
            None,
            "files",
            0.0,
            None,
            Timestamp("2023-12-01 00:00:00"),
            None,
            [Path(r"cool"), Path(r"fun")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"cool")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01"),
                        end=Timestamp("2023-12-01 00:00:02"),
                    ),
                    [Path(r"fun")],
                ),
            ],
            id="multiple_files_no_strptime_should_be_consecutive",
        ),
        pytest.param(
            None,
            None,
            None,
            None,
            "files",
            0.0,
            None,
            Timestamp("2023-12-01 00:00:00"),
            None,
            [Path(r"cool"), Path("boring.shenanigan")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"cool")],
                ),
            ],
            id="only_specified_formats_are_kept",
        ),
        pytest.param(
            None,
            None,
            None,
            None,
            "timedelta_total",
            0.0,
            Timedelta(seconds=0.5),
            Timestamp("2023-12-01 00:00:00"),
            None,
            [Path(r"cool"), Path(r"fun")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00"),
                        end=Timestamp("2023-12-01 00:00:00.5"),
                    ),
                    [Path(r"cool")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00.5"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"cool")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01"),
                        end=Timestamp("2023-12-01 00:00:01.5"),
                    ),
                    [Path(r"fun")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01.5"),
                        end=Timestamp("2023-12-01 00:00:02"),
                    ),
                    [Path(r"fun")],
                ),
            ],
            id="timedelta_total",
        ),
        pytest.param(
            None,
            None,
            None,
            None,
            "timedelta_total",
            0.5,
            Timedelta(seconds=1),
            Timestamp("2023-12-01 00:00:00"),
            None,
            [Path(r"cool"), Path(r"fun")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"cool")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00.5"),
                        end=Timestamp("2023-12-01 00:00:01.5"),
                    ),
                    [Path(r"cool"), Path(r"fun")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01"),
                        end=Timestamp("2023-12-01 00:00:02"),
                    ),
                    [Path(r"fun")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01.5"),
                        end=Timestamp("2023-12-01 00:00:02.5"),
                    ),
                    [Path(r"fun")],
                ),
            ],
            id="timedelta_total_with_overlap",
        ),
        pytest.param(
            None,
            Timestamp("2023-12-01 00:00:00.5"),
            Timestamp("2023-12-01 00:00:01.5"),
            None,
            "timedelta_total",
            0.0,
            Timedelta(seconds=0.5),
            Timestamp("2023-12-01 00:00:00"),
            None,
            [Path(r"cool"), Path(r"fun")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00.5"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"cool")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01"),
                        end=Timestamp("2023-12-01 00:00:01.5"),
                    ),
                    [Path(r"fun")],
                ),
            ],
            id="timedelta_total_between_timestamps",
        ),
        pytest.param(
            "%y%m%d%H%M%S",
            Timestamp("2023-12-01 00:00:00.5"),
            Timestamp("2023-12-01 00:00:01.5"),
            None,
            "timedelta_total",
            0.0,
            Timedelta(seconds=0.5),
            None,
            None,
            [Path(r"231201000000"), Path(r"231201000001")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00.5"),
                        end=Timestamp("2023-12-01 00:00:01"),
                    ),
                    [Path(r"231201000000")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01"),
                        end=Timestamp("2023-12-01 00:00:01.5"),
                    ),
                    [Path(r"231201000001")],
                ),
            ],
            id="striptime_format",
        ),
        pytest.param(
            "%y%m%d%H%M%S",
            Timestamp("2023-12-01 00:00:00.5+01:00"),
            Timestamp("2023-12-01 00:00:01.5+01:00"),
            "Europe/Warsaw",
            "timedelta_total",
            0.0,
            Timedelta(seconds=0.5),
            None,
            None,
            [Path(r"231201000000"), Path(r"231201000001")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:00.5+01:00"),
                        end=Timestamp("2023-12-01 00:00:01+01:00"),
                    ),
                    [Path(r"231201000000")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 00:00:01+01:00"),
                        end=Timestamp("2023-12-01 00:00:01.5+01:00"),
                    ),
                    [Path(r"231201000001")],
                ),
            ],
            id="timezone_location",
        ),
        pytest.param(
            "%y%m%d%H%M%S%z",
            Timestamp("2023-12-01 01:00:00.5+01:00"),
            Timestamp("2023-12-01 01:00:01.5+01:00"),
            "Europe/Warsaw",
            "timedelta_total",
            0.0,
            Timedelta(seconds=0.5),
            None,
            None,
            [Path(r"231201000000+0000"), Path(r"231201000001+0000")],
            [
                (
                    Event(
                        begin=Timestamp("2023-12-01 01:00:00.5+01:00"),
                        end=Timestamp("2023-12-01 01:00:01+01:00"),
                    ),
                    [Path(r"231201000000+0000")],
                ),
                (
                    Event(
                        begin=Timestamp("2023-12-01 01:00:01+01:00"),
                        end=Timestamp("2023-12-01 01:00:01.5+01:00"),
                    ),
                    [Path(r"231201000001+0000")],
                ),
            ],
            id="timezone_conversion",
        ),
    ],
)
def test_base_dataset_from_folder(
    monkeypatch: pytest.monkeypatch,
    strptime_format: str | None,
    begin: Timestamp | None,
    end: Timestamp | None,
    timezone: str | None,
    mode: Literal["files", "timedelta_total", "timedelta_file"],
    overlap: float,
    data_duration: Timedelta | None,
    first_file_begin: Timestamp | None,
    name: str | None,
    files: list[Path],
    expected_data_events: list[tuple[Event, list[Path]]],
) -> None:
    monkeypatch.setattr(Path, "iterdir", lambda x: files)

    bds = DummyDataset.from_folder(
        folder=Path("foo"),
        strptime_format=strptime_format,
        begin=begin,
        end=end,
        timezone=timezone,
        mode=mode,
        overlap=overlap,
        data_duration=data_duration,
        first_file_begin=first_file_begin,
        name=name,
    )

    assert bds.name == name if name else str(bds.begin)

    for expected, data in zip(
        sorted(expected_data_events, key=lambda e: e[0].begin),
        sorted(bds.data, key=lambda e: e.begin),
        strict=True,
    ):
        assert data.begin == expected[0].begin
        assert data.end == expected[0].end
        assert np.array_equal(sorted(f.path for f in data.files), sorted(expected[1]))


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
    bf = DummyFile(
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
    dummy_dataset: DummyDataset,
) -> None:
    origin_files = [Path(str(file.path)) for file in dummy_dataset.files]

    # The starting folder of the dataset is the folder where the files are located
    assert dummy_dataset.folder == tmp_path

    destination = tmp_path / "destination"
    dummy_dataset.folder = destination

    # Setting the folder shouldn't move the files
    assert all(file.path.parent == tmp_path for file in dummy_dataset.files)
    assert all(file.exists for file in origin_files)

    # Folder should be changed when dataset is moved
    new_destination = tmp_path / "new_destination"
    dummy_dataset.move_files(new_destination)

    assert new_destination.exists()
    assert new_destination.is_dir()
    assert dummy_dataset.folder == new_destination
    assert all((new_destination / file.name).exists() for file in origin_files)
    assert not any(file.exists() for file in origin_files)


@pytest.mark.parametrize(
    ("files", "mode", "data_duration", "expected_data"),
    [
        pytest.param(
            [
                DummyFile(
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
                DummyFile(
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
                DummyFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                DummyFile(
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
                DummyFile(
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
                DummyFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                DummyFile(
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
                DummyFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                DummyFile(
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
    files: list[DummyFile],
    mode: Literal["files", "timedelta_total"],
    data_duration: Timedelta | None,
    expected_data: list[tuple[Event, str]],
) -> None:
    ds = DummyDataset.from_files(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
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
                DummyFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                DummyFile(
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
                DummyFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                DummyFile(
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
                DummyFile(
                    path=Path("cool"),
                    begin=Timestamp("2022-04-22 12:12:12"),
                    end=Timestamp("2022-04-22 12:12:13"),
                ),
                DummyFile(
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
    files: list[DummyFile],
    begin: Timestamp,
    end: Timestamp,
    expected_data: Event,
) -> None:
    data = DummyData.from_files(files=files)
    if begin:
        data.begin = begin
    if end:
        data.end = end
    assert data.begin == expected_data.begin
    assert data.end == expected_data.end

    def mocked_get_value(self: DummyData) -> None:
        for item in data.items:
            if item.is_empty:
                continue
            assert item.file.begin <= item.begin
            assert item.file.end >= item.end

    monkeypatch.setattr(DummyData, "get_value", mocked_get_value)

    data.get_value()


@pytest.mark.parametrize(
    ("data1", "data2", "expected"),
    [
        pytest.param(
            DummyData.from_files(
                [
                    DummyFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    DummyFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:14"),
                        end=Timestamp("2015-08-28 12:13:15"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    DummyFile(
                        "house",
                        begin=Timestamp("2015-08-28 12:12:14"),
                        end=Timestamp("2015-08-28 12:13:15"),
                    ),
                ],
                begin=None,
                end=None,
            ),
            DummyData.from_files(
                [
                    DummyFile(
                        "cherry",
                        begin=Timestamp("2015-08-28 12:12:12"),
                        end=Timestamp("2015-08-28 12:13:12"),
                    ),
                    DummyFile(
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
def test_base_data_equality(data1: DummyData, data2: DummyData, expected: bool) -> None:
    assert (data1 == data2) == expected


@pytest.mark.parametrize(
    ("data", "name", "expected"),
    [
        pytest.param(
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
                        "beach",
                        begin=Timestamp("2015-08-28 12:13:12"),
                        end=Timestamp("2015-08-28 12:14:12"),
                    ),
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
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
            DummyData.from_files(
                [
                    DummyFile(
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
def test_data_name(data: DummyData, name: str | None, expected: str) -> None:
    data.name = name
    assert data.name == expected
    assert str(data) == expected


@pytest.mark.parametrize(
    ("files", "begin", "end", "data_duration", "mode", "overlap", "expected"),
    [
        pytest.param(
            [
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                DummyFile(
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
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:12"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:22"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:22"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:12"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:00"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                DummyFile(
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
                DummyFile(
                    "depression",
                    begin=Timestamp("2015-08-28 12:12:00"),
                    end=Timestamp("2015-08-28 12:13:02"),
                ),
                DummyFile(
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
    files: list[DummyFile],
    begin: Timestamp,
    end: Timestamp,
    data_duration: Timedelta,
    mode: Literal["timedelta_total", "timedelta_file"],
    overlap: float,
    expected: list[list[tuple[Event, str | None]]],
) -> None:
    data = DummyDataset.from_files(
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


def test_dummydata_make_split_data() -> None:
    dfs = [
        DummyFile(
            path=Path("foo"),
            begin=Timestamp("2020-01-01 00:00:00"),
            end=Timestamp("2020-01-01 00:01:00"),
        ),
        DummyFile(
            path=Path("bar"),
            begin=Timestamp("2020-01-01 00:01:00"),
            end=Timestamp("2020-01-01 00:02:00"),
        ),
    ]
    dd = DummyData.from_files(dfs)
    split_dd = dd._make_split_data(
        files=dfs,
        begin=Timestamp("2020-01-01 00:00:30"),
        end=Timestamp("2020-01-01 00:01:30"),
        name="cool",
    )
    assert split_dd == DummyData.from_files(
        files=dfs,
        begin=Timestamp("2020-01-01 00:00:30"),
        end=Timestamp("2020-01-01 00:01:30"),
        name="cool",
    )
    assert np.array_equal(
        dd.split(),
        [
            DummyData.from_files([dfs[0]]),
            DummyData.from_files([dfs[1]]),
        ],
    )


def test_dummydata_make_file() -> None:
    dfs = [
        DummyFile(
            path=Path("foo"),
            begin=Timestamp("2020-01-01 00:00:00"),
            end=Timestamp("2020-01-01 00:0:01"),
        ),
        DummyFile(
            path=Path("bar"),
            begin=Timestamp("2020-01-01 00:01:00"),
            end=Timestamp("2020-01-01 00:02:00"),
        ),
    ]
    dd = DummyData.from_files(dfs)
    assert dd._make_file(Path("foo"), begin=Timestamp("2020-01-01 00:00:00")) == dfs[0]


def test_dummydata_from_base_dict() -> None:
    dfs = [
        DummyFile(
            path=Path("foo"),
            begin=Timestamp("2020-01-01 00:00:00"),
            end=Timestamp("2020-01-01 00:0:01"),
        ),
        DummyFile(
            path=Path("bar"),
            begin=Timestamp("2020-01-01 00:02:00"),
            end=Timestamp("2020-01-01 00:03:00"),
        ),
    ]
    dd = DummyData.from_files(
        files=dfs,
        begin=Timestamp("2020-01-01 00:00:30"),
        end=Timestamp("2020-01-01 00:02:30"),
        name="cool",
    )
    dictionary = {}
    assert (
        dd._from_base_dict(
            dictionary=dictionary,
            files=dfs,
            begin=dd.begin,
            end=dd.end,
            name="cool",
        )
        == dd
    )


def test_dummydataset_data_from_dict() -> None:
    dfs = [
        DummyFile(
            path=Path("foo"),
            begin=Timestamp("2020-01-01 00:00:00"),
            end=Timestamp("2020-01-01 00:0:01"),
        ),
        DummyFile(
            path=Path("bar"),
            begin=Timestamp("2020-01-01 00:00:02"),
            end=Timestamp("2020-01-01 00:00:03"),
        ),
    ]
    dd1 = DummyData.from_files(
        files=dfs,
        begin=Timestamp("2020-01-01 00:00:00.5"),
        end=Timestamp("2020-01-01 00:00:01.5"),
        name="cool",
    )
    dd2 = DummyData.from_files(
        files=dfs,
        begin=Timestamp("2020-01-01 00:00:01.75"),
        end=Timestamp("2020-01-01 00:00:02.25"),
        name="fun",
    )
    dds = DummyDataset(data=[dd1, dd2])
    dictionary = {"data": dd1.to_dict()}
    assert (
        dds._data_from_dict(
            dictionary=dictionary,
        )[0]
        == dd1
    )
