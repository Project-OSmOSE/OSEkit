from __future__ import annotations

from typing import Generic, TypeVar

from pandas import Timedelta, Timestamp, date_range

from OSmOSE.data.data_base import DataBase
from OSmOSE.data.file_base import FileBase

TData = TypeVar("TData", bound=DataBase)
TFile = TypeVar("TFile", bound=FileBase)


class DatasetBase(Generic[TData, TFile]):
    def __init__(self, data: list[TData]):
        self.data = data

    @property
    def begin(self):
        return min(data.begin for data in self.data)

    @property
    def end(self):
        return max(data.end for data in self.data)

    @classmethod
    def data_from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> list[DataBase]:
        return [
            DataBase.from_files(files, begin=b, end=b + data_duration)
            for b in date_range(begin, end, freq=data_duration)[:-1]
        ]
