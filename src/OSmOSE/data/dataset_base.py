from __future__ import annotations

from typing import Generic, TypeVar

from pandas import Timedelta, Timestamp, date_range

from OSmOSE.data.data_base import DataBase
from OSmOSE.data.file_base import FileBase

TData = TypeVar("TData", bound=DataBase)
TFile = TypeVar("TFile", bound=FileBase)


class DatasetBase(Generic[TData, TFile]):
    """Base class for Dataset objects.

    Datasets are collections of Data, with methods
    that simplify repeated operations on the data.
    """

    def __init__(self, data: list[TData]) -> None:
        """Instantiate a Dataset object from the Data objects."""
        self.data = data

    @property
    def begin(self) -> Timestamp:
        """Begin of the first data object."""
        return min(data.begin for data in self.data)

    @property
    def end(self) -> Timestamp:
        """End of the last data object."""
        return max(data.end for data in self.data)

    @classmethod
    def data_from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> list[DataBase]:
        """Return a list of DataBase objects from File objects.

        These DataBase are linked to the file through ItemBase objects.
        Specialized Dataset classes can use these DataBase objects parameters to
        instantiate specialized Data objects.

        Parameters
        ----------
        files: list[TFile]
            The list of files from which the Data objects are built.
        begin: Timestamp | None
            Begin of the first data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the last data object.
            Defaulted to the end of the last file.
        data_duration: Timedelta | None
            Duration of the data objects.
            If provided, data will be evenly distributed between begin and end.
            Else, one data object will cover the whole period.

        Returns
        -------
        list[DataBase]:
            A list of DataBase objects.

        """
        return [
            DataBase.from_files(files, begin=b, end=b + data_duration)
            for b in date_range(begin, end, freq=data_duration)[:-1]
        ]

    @classmethod
    def from_files(
        cls,
        files: list[TFile],
        begin: Timestamp,
        end: Timestamp,
        data_duration: Timedelta,
    ) -> DatasetBase:
        data_base = [
            DataBase.from_files(files, begin=b, end=b + data_duration)
            for b in date_range(begin, end, freq=data_duration)[:-1]
        ]
        return cls(data_base)
