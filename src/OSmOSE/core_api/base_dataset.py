"""BaseDataset: Base class for the Dataset objects.

Datasets are collections of Data, with methods
that simplify repeated operations on the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from pandas import Timedelta, Timestamp, date_range

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.core_api.base_data import BaseData
from OSmOSE.core_api.base_file import BaseFile
from OSmOSE.core_api.event import Event
from OSmOSE.core_api.json_serializer import deserialize_json, serialize_json

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

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)

    @property
    def begin(self) -> Timestamp:
        """Begin of the first data object."""
        return min(data.begin for data in self.data)

    @property
    def end(self) -> Timestamp:
        """End of the last data object."""
        return max(data.end for data in self.data)

    @property
    def files(self) -> set[TFile]:
        """All files referred to by the Dataset."""
        return {file for data in self.data for file in data.files}

    def write(self, folder: Path) -> None:
        """Write all data objects in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which to write the data.

        """
        for data in self.data:
            data.write(folder)

    def to_dict(self) -> dict:
        """Serialize a BaseDataset to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the BaseDataset.

        """
        return {str(d): d.to_dict() for d in self.data}

    @classmethod
    def from_dict(cls, dictionary: dict) -> BaseDataset:
        """Deserialize a BaseDataset from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the BaseData.

        Returns
        -------
        AudioData
            The deserialized BaseDataset.

        """
        return cls([BaseData.from_dict(d) for d in dictionary.values()])

    def write_json(self, folder: Path) -> None:
        """Write a serialized BaseDataset to a JSON file."""
        serialize_json(folder / f"{self}.json", self.to_dict())

    @classmethod
    def from_json(cls, file: Path) -> BaseDataset:
        """Deserialize a BaseDataset from a JSON file.

        Parameters
        ----------
        file: Path
            Path to the serialized JSON file representing the BaseDataset.

        Returns
        -------
        BaseDataset
            The deserialized BaseDataset.

        """
        return cls.from_dict(deserialize_json(file))

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
                for b in date_range(begin, end, freq=data_duration, inclusive="left")
            ]
        else:
            data_base = [BaseData.from_files(files, begin=begin, end=end)]
        return cls(data_base)
