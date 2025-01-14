"""BaseDataset: Base class for the Dataset objects.

Datasets are collections of Data, with methods
that simplify repeated operations on the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from pandas import Timedelta, Timestamp, date_range

from OSmOSE.data.base_data import BaseData
from OSmOSE.data.base_file import BaseFile
from OSmOSE.data.event import Event

if TYPE_CHECKING:
    from pathlib import Path

TData = TypeVar("TData", bound=BaseData)
TFile = TypeVar("TFile", bound=BaseFile)


class BaseDataset(Generic[TData, TFile], Event):
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

    def write(self, folder: Path) -> None:
        """Write all data objects in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which to write the data.

        """
        for data in self.data:
            data.write(folder)

    @classmethod
    def from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> BaseDataset:
        """Return a base BaseDataset object from a list of Files.

        Parameters
        ----------
        files: list[TFile]
            The list of files contained in the Dataset.
        begin: Timestamp | None
            Begin of the first data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the last data object.
            Defaulted to the end of the last file.
        data_duration: Timedelta | None
            Duration of the data objects.
            If provided, data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.

        Returns
        -------
        BaseDataset[TItem, TFile]:
        The DataBase object.

        """
        if not begin:
            begin = min(file.begin for file in files)
        if not end:
            end = max(file.end for file in files)
        if data_duration:
            data_base = [
                BaseData.from_files(files, begin=b, end=b + data_duration)
                for b in date_range(begin, end, freq=data_duration)[:-1]
            ]
        else:
            data_base = [BaseData.from_files(files, begin=begin, end=end)]
        return cls(data_base)
