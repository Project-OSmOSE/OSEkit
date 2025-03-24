"""BaseDataset: Base class for the Dataset objects.

Datasets are collections of Data, with methods
that simplify repeated operations on the data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Literal, TypeVar

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

    import pytz

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

    def __eq__(self, other: BaseDataset) -> bool:
        """Overwrite __eq__."""
        return sorted(self.data, key=lambda e: (e.begin, e.end)) == sorted(
            other.data, key=lambda e: (e.begin, e.end)
        )

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

    @property
    def folder(self) -> Path:
        """Folder in which the dataset files are located."""
        return next(iter(file.path.parent for file in self.files), None)

    @folder.setter
    def folder(self, folder: Path) -> None:
        """Move the dataset to the specified destination folder.

        Parameters
        ----------
        folder: Path
            The folder in which the dataset will be moved.
            It will be created if it does not exist.

        """
        for file in self.files:
            file.move(folder)

    @property
    def serialized_file(self) -> Path:
        """Return the path of the serialized file of this dataset."""
        return self.folder / f"{self}.json"

    @property
    def data_duration(self) -> Timedelta:
        """Return the most frequent duration among durations of the data of this dataset, rounded to the nearest second."""
        data_durations = [
            Timedelta(data.duration).round(freq="1s") for data in self.data
        ]
        return max(set(data_durations), key=data_durations.count)

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
        bound: Literal["files", "timedelta"] = "timedelta",
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
        bound: Literal["files", "timedelta"]
            Bound between the original files and the dataset data.
            "files": one data will be created for each file.
            "timedelta": data objects of duration equal to data_duration will
            be created.
        data_duration: Timedelta | None
            Duration of the data objects.
            If bound is set to "files", this parameter has no effect.
            If provided, data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.

        Returns
        -------
        BaseDataset[TItem, TFile]:
        The DataBase object.

        """
        if bound == "files":
            data_base = [BaseData.from_files([f]) for f in files]
            data_base = BaseData.remove_overlaps(data_base)
            return cls(data_base)

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

    @classmethod
    def from_folder(  # noqa: PLR0913
        cls,
        folder: Path,
        strptime_format: str,
        file_class: type[TFile] = BaseFile,
        supported_file_extensions: list[str] | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        timezone: str | pytz.timezone | None = None,
        bound: Literal["files", "timedelta"] = "timedelta",
        data_duration: Timedelta | None = None,
    ) -> BaseDataset:
        """Return a BaseDataset from a folder containing the base files.

        Parameters
        ----------
        folder: Path
            The folder containing the files.
        strptime_format: str
            The strptime format of the timestamps in the file names.
        file_class: type[Tfile]
            Derived type of BaseFile used to instantiate the dataset.
        supported_file_extensions: list[str]
            List of supported file extensions for parsing TFiles.
        begin: Timestamp | None
            The begin of the dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the dataset.
            Defaulted to the end of the last file.
        timezone: str | pytz.timezone | None
            The timezone in which the file should be localized.
            If None, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.
        bound: Literal["files", "timedelta"]
            Bound between the original files and the dataset data.
            "files": one data will be created for each file.
            "timedelta": data objects of duration equal to data_duration will
            be created.
        data_duration: Timedelta | None
            Duration of the data objects.
            If bound is set to "files", this parameter has no effect.
            If provided, data will be evenly distributed between begin and end.
            Else, one object will cover the whole time period.

        Returns
        -------
        Basedataset:
            The base dataset.

        """
        if supported_file_extensions is None:
            supported_file_extensions = []
        valid_files = []
        rejected_files = []
        for file in folder.iterdir():
            if file.suffix.lower() not in supported_file_extensions:
                continue
            try:
                f = file_class(file, strptime_format=strptime_format, timezone=timezone)
                valid_files.append(f)
            except (ValueError, LibsndfileError):
                rejected_files.append(file)

        if rejected_files:
            rejected_files = "\n\t".join(f.name for f in rejected_files)
            glc.logger.warn(
                f"The following files couldn't be parsed:\n\t{rejected_files}",
            )

        if not valid_files:
            raise FileNotFoundError(f"No valid file found in {folder}.")

        return BaseDataset.from_files(valid_files, begin, end, bound, data_duration)
