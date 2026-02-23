import typing
from pathlib import Path
from typing import Self

import numpy as np
from pandas import Timestamp

from osekit.core_api.base_data import BaseData, TFile
from osekit.core_api.base_dataset import BaseDataset, TData
from osekit.core_api.base_file import BaseFile
from osekit.core_api.base_item import BaseItem


class DummyFile(BaseFile):
    supported_extensions: typing.ClassVar = [""]

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray: ...


class DummyItem(BaseItem[DummyFile]): ...


class DummyData(BaseData[DummyItem, DummyFile]):
    item_cls = DummyItem

    def write(self, folder: Path, *, link: bool = False) -> None: ...

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
