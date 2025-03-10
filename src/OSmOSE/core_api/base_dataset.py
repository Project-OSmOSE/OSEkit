"""BaseDataset: Base class for the Dataset objects.

Datasets are collections of Data, with methods
that simplify repeated operations on the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from pandas import Timedelta, Timestamp, date_range
from soundfile import LibsndfileError

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.config import global_logging_context as glc
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
                for b in date_range(begin, end, freq=data_duration)[:-1]
            ]
        else:
            data_base = [BaseData.from_files(files, begin=begin, end=end)]
        return cls(data_base)

    @classmethod
    def from_folder(  # noqa: PLR0913
        cls,
        folder: Path,
        strptime_format: str,
        file_class: type[TFile] = BaseFile,
        supported_file_extensions: list[str] = [],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> BaseDataset:
        """Return a BaseDataset from a folder containing the base files.

        Parameters
        ----------
        folder: Path
            The folder containing the audio files.
        strptime_format: str
            The strptime format of the timestamps in the audio file names.
        file_class: type[Tfile]
            Derived type of BaseFile used to instantiate the dataset.
        supported_file_extensions: list[str]
            List of supported file extensions for parsing TFiles.
        begin: Timestamp | None
            The begin of the audio dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the audio dataset.
            Defaulted to the end of the last file.
        data_duration: Timedelta | None
            Duration of the audio data objects.
            If provided, audio data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.

        Returns
        -------
        Basedataset:
            The base dataset.

        """
        valid_files = []
        rejected_files = []
        for file in folder.iterdir():
            if file.suffix.lower() not in supported_file_extensions:
                continue
            try:
                f = file_class(file, strptime_format=strptime_format)
                valid_files.append(f)
            except (ValueError, LibsndfileError):
                rejected_files.append(file)

        if rejected_files:
            rejected_files = "\n\t".join(f.name for f in rejected_files)
            glc.logger.warn(
                f"The following files couldn't be parsed:\n\t{rejected_files}",
            )

        if not valid_files:
            raise FileNotFoundError(f"No valid audio file found in {folder}.")

        return BaseDataset.from_files(valid_files, begin, end, data_duration)
