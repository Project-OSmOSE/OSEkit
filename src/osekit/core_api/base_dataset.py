"""``BaseDataset``: Base class for the Dataset objects.

Datasets are collections of Data, with methods
that simplify repeated operations on the data.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from bisect import bisect
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TypeVar

from pandas import Timedelta, Timestamp, date_range
from soundfile import LibsndfileError
from tqdm import tqdm

from osekit.config import TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED
from osekit.config import global_logging_context as glc
from osekit.core_api.base_data import BaseData
from osekit.core_api.base_file import BaseFile
from osekit.core_api.event import Event
from osekit.core_api.json_serializer import deserialize_json, serialize_json
from osekit.utils.timestamp_utils import last_window_end

if TYPE_CHECKING:
    import pytz

TData = TypeVar("TData", bound=BaseData)
TFile = TypeVar("TFile", bound=BaseFile)


class BaseDataset[TData: BaseData, TFile: BaseFile](Event, ABC):
    """Base class for Dataset objects.

    Datasets are collections of Data, with methods
    that simplify repeated operations on the data.
    """

    file_cls: type[TFile]

    def __init__(
        self,
        data: list[TData],
        name: str | None = None,
        suffix: str = "",
        folder: Path | None = None,
    ) -> None:
        """Instantiate a Dataset object from the Data objects."""
        self.data = data
        self._name = name
        self._has_default_name = name is None
        self._suffix = suffix
        self._folder = folder

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.name

    def __eq__(self, other: BaseDataset) -> bool:
        """Overwrite __eq__."""
        return sorted(self.data, key=lambda e: (e.begin, e.end)) == sorted(
            other.data,
            key=lambda e: (e.begin, e.end),
        )

    @property
    def base_name(self) -> str:
        """Name of the dataset without suffix."""
        return (
            self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_UNLOCALIZED)
            if self._name is None
            else self._name
        )

    @base_name.setter
    def base_name(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        """Name of the dataset with suffix."""
        return self.base_name if not self.suffix else f"{self.base_name}_{self.suffix}"

    @name.setter
    def name(self, name: str | None) -> None:
        self._name = name

    @property
    def suffix(self) -> str:
        """Suffix that is applied to the name of the ads.

        This is used by the public API, for suffixing multiple core_api datasets
        that are created simultaneously and share the same namewith their specific type,
        e.g. ``_audio`` or ``_spectro``.
        """
        return self._suffix

    @suffix.setter
    def suffix(self, suffix: str | None) -> None:
        self._suffix = suffix

    @property
    def has_default_name(self) -> bool:
        """Return ``True`` if the dataset has a default name, ``False`` if it has a given name."""
        return self._has_default_name

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
        """Folder in which the dataset files are located or to be written."""
        return (
            self._folder
            if self._folder is not None
            else next(iter(file.path.parent for file in self.files), None)
        )

    @folder.setter
    def folder(self, folder: Path) -> None:
        """Set the folder in which the dataset files might be written.

        Parameters
        ----------
        folder: Path
            The folder in which the dataset files might be written.

        """
        self._folder = folder

    def move_files(self, folder: Path) -> None:
        """Move the dataset files to the destination folder.

        Parameters
        ----------
        folder: Path
            Destination folder in which the dataset files will be moved.

        """
        for file in tqdm(
            self.files,
            disable=os.getenv("DISABLE_TQDM", "False").lower() in ("true", "1", "t"),
        ):
            file.move(folder)
        self._folder = folder

    @property
    def data_duration(self) -> Timedelta:
        """Return the most frequent duration among the data of this dataset.

        The duration is rounded to the nearest second.

        """
        data_durations = [
            Timedelta(data.duration).round(freq="1s") for data in self.data
        ]
        return max(set(data_durations), key=data_durations.count)

    def write(
        self,
        folder: Path,
        first: int = 0,
        last: int | None = None,
        *,
        link: bool = False,
    ) -> None:
        """Write all data objects in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which to write the data.
        link: bool
            If ``True``, the Data will be bound to the written file.
            Its items will be replaced with a single item, which will match the whole
            new File.
        first: int
            Index of the first data object to write.
        last: int | None
            Index after the last data object to write.

        """
        last = len(self.data) if last is None else last
        for data in tqdm(
            self.data[first:last],
            disable=os.getenv("DISABLE_TQDM", "False").lower() in ("true", "1", "t"),
        ):
            data.write(folder=folder, link=link)

    def to_dict(self) -> dict:
        """Serialize a ``BaseDataset`` to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the ``BaseDataset``.

        """
        return {
            "data": {str(d): d.to_dict() for d in self.data},
            "name": self._name,
            "suffix": self.suffix,
            "folder": str(self.folder),
        }

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self:
        """Deserialize a ``BaseDataset`` from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the ``BaseDataset``.

        Returns
        -------
        AudioData
            The deserialized ``BaseDataset``.

        """
        data = cls._data_from_dict(dictionary["data"])
        name = dictionary["name"]
        suffix = dictionary["suffix"]
        folder = Path(dictionary["folder"])
        return cls(data=data, name=name, suffix=suffix, folder=folder)

    @classmethod
    @abstractmethod
    def _data_from_dict(cls, dictionary: dict) -> list[TData]:
        """Return a list of Data from a serialized dictionary."""
        ...

    def write_json(self, folder: Path) -> None:
        """Write a serialized ``BaseDataset`` to a JSON file."""
        serialize_json(folder / f"{self.name}.json", self.to_dict())

    @classmethod
    def from_json(cls, file: Path) -> Self:
        """Deserialize a ``BaseDataset`` from a JSON file.

        Parameters
        ----------
        file: Path
            Path to the serialized JSON file representing the ``BaseDataset``.

        Returns
        -------
        BaseDataset
            The deserialized ``BaseDataset``.

        """
        return cls.from_dict(deserialize_json(file))

    @classmethod
    def from_files(  # noqa: PLR0913
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        data_duration: Timedelta | None = None,
        overlap: float = 0.0,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        """Return a Dataset object from a list of Files.

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
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration`` will
            be created from the ``begin`` timestamp to the ``end`` timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration`` will
            be created from the beginning of the first file that the ``begin`` timestamp
            is into, until it would resume in a data beginning between two files.
            Then, the next data object will be created from the
            beginning of the next original file and so on.
        data_duration: Timedelta | None
            Duration of the data objects.
            If mode is set to ``"files"``, this parameter has no effect.
            If provided, data will be evenly distributed between ``begin`` and ``end``.
            Else, one data object will cover the whole time period.
        overlap: float
            Overlap percentage between consecutive data.
        name: str|None
            Name of the dataset.
        kwargs:
            Keyword arguments to pass to the ``cls.data_from_files()`` method.

        Returns
        -------
        Self:
            The Dataset object.

        """
        if mode == "files":
            data = [cls._data_from_files([f], **kwargs) for f in files]
            data = BaseData.remove_overlaps(data)
            return cls(data=data, name=name)

        if not begin:
            begin = min(file.begin for file in files)
        if not end:
            end = max(file.end for file in files)
        if data_duration:
            data_base = (
                cls._get_data_from_files_timedelta_total(
                    begin=begin,
                    end=end,
                    data_duration=data_duration,
                    files=files,
                    overlap=overlap,
                    **kwargs,
                )
                if mode == "timedelta_total"
                else cls._get_data_from_files_timedelta_file(
                    begin=begin,
                    end=end,
                    data_duration=data_duration,
                    files=files,
                    overlap=overlap,
                    **kwargs,
                )
            )
        else:
            data_base = [
                cls._data_from_files(files=files, begin=begin, end=end, **kwargs),
            ]
        return cls(data_base, name=name)

    @classmethod
    @abstractmethod
    def _data_from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> TData:
        """Return a base Dataset object between two timestamps from a list of Files."""
        ...

    @classmethod
    def _get_data_from_files_timedelta_total(
        cls,
        begin: Timestamp,
        end: Timestamp,
        data_duration: Timedelta,
        files: list[TFile],
        overlap: float = 0,
        **kwargs,  # noqa: ANN003
    ) -> list[TData]:
        if not 0 <= overlap < 1:
            msg = f"Overlap ({overlap}) must be between 0 and 1."
            raise ValueError(msg)

        active_file_index = 0
        output = []
        files = sorted(files, key=lambda f: f.begin)
        freq = data_duration * (1 - overlap)

        for data_begin in tqdm(
            date_range(begin, end, freq=freq, inclusive="left"),
            disable=os.getenv("DISABLE_TQDM", "False").lower() in ("true", "1", "t"),
        ):
            data_end = Timestamp(data_begin + data_duration)
            while (
                active_file_index < len(files)
                and files[active_file_index].end < data_begin
            ):
                active_file_index += 1
            last_active_file_index = active_file_index
            while (
                last_active_file_index < len(files)
                and files[last_active_file_index].begin < data_end
            ):
                last_active_file_index += 1
            output.append(
                cls._data_from_files(
                    files[active_file_index:last_active_file_index],
                    data_begin,
                    data_end,
                    **kwargs,
                ),
            )

        return output

    @classmethod
    def _get_data_from_files_timedelta_file(
        cls,
        begin: Timestamp,
        end: Timestamp,
        data_duration: Timedelta,
        files: list[TFile],
        overlap: float = 0,
        **kwargs,
    ) -> list[TData]:
        if not 0 <= overlap < 1:
            msg = f"Overlap ({overlap}) must be between 0 and 1."
            raise ValueError(msg)

        files = sorted(files, key=lambda file: file.begin)
        first = max(0, bisect(files, begin, key=lambda f: f.begin) - 1)
        last = bisect(files, end, key=lambda f: f.begin)

        data_hop = data_duration * (1 - overlap)

        output = []
        files_chunk = []
        for idx, file in tqdm(
            enumerate(files[first:last]),
            disable=os.getenv("DISABLE_TQDM", "False").lower() in ("true", "1", "t"),
        ):
            if file in files_chunk:
                continue
            files_chunk = [file]

            for next_file in files[idx + 1 :]:
                upper_data_limit = last_window_end(
                    begin=file.begin,
                    end=files_chunk[-1].end,
                    window_hop=data_hop,
                    window_duration=data_duration,
                )
                if upper_data_limit < next_file.begin:
                    break
                files_chunk.append(next_file)

            output.extend(
                cls._data_from_files(
                    files,
                    data_begin,
                    data_begin + data_duration,
                    **kwargs,
                )
                for data_begin in date_range(
                    file.begin,
                    files_chunk[-1].end,
                    freq=data_hop,
                    inclusive="left",
                )
            )

        return output

    @classmethod
    def from_folder(  # noqa: PLR0913
        cls: type[Self],
        folder: Path,
        strptime_format: str | None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        timezone: str | pytz.timezone | None = None,
        mode: Literal["files", "timedelta_total", "timedelta_file"] = "timedelta_total",
        overlap: float = 0.0,
        data_duration: Timedelta | None = None,
        first_file_begin: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> Self:
        """Return a Dataset from a folder containing the base files.

        Parameters
        ----------
        folder: Path
            The folder containing the files.
        strptime_format: str | None
            The strptime format used in the filenames.
            It should use valid strftime codes (https://strftime.org/).
            If None, the first audio file of the folder will start
            at ``first_file_begin``, and each following file will start
            at the end of the previous one.
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
        mode: Literal["files", "timedelta_total", "timedelta_file"]
            Mode of creation of the dataset data from the original files.
            ``"files"``: one data will be created for each file.
            ``"timedelta_total"``: data objects of duration equal to ``data_duration`` will
            be created from the ``begin`` timestamp to the ``end`` timestamp.
            ``"timedelta_file"``: data objects of duration equal to ``data_duration`` will
            be created from the beginning of the first file that the ``begin`` timestamp is into, until it would resume
            in a data beginning between two files. Then, the next data object will be created from the
            beginning of the next original file and so on.
        overlap: float
            Overlap percentage between consecutive data.
        data_duration: Timedelta | None
            Duration of the data objects.
            If mode is set to ``"files"``, this parameter has no effect.
            If provided, data will be evenly distributed between ``begin`` and ``end``.
            Else, one object will cover the whole time period.
        first_file_begin: Timestamp | None
            Timestamp of the first audio file being processed.
            Will be ignored if ``striptime_format`` is specified.
        name: str|None
            Name of the dataset.
        kwargs:
            Keyword arguments to pass to the ``cls.from_files()`` method.

        Returns
        -------
        Self:
            The dataset.

        """
        valid_files = []
        rejected_files = []
        first_file_begin = first_file_begin or Timestamp("2020-01-01 00:00:00")
        for file in tqdm(
            sorted(folder.iterdir()),
            disable=os.getenv("DISABLE_TQDM", "False").lower() in ("true", "1", "t"),
        ):
            is_file_ok = cls._parse_file(
                file=file,
                strptime_format=strptime_format,
                timezone=timezone,
                begin_timestamp=first_file_begin,
                valid_files=valid_files,
                rejected_files=rejected_files,
            )
            if is_file_ok:
                first_file_begin += valid_files[-1].duration

        if rejected_files:
            rejected_files = "\n\t".join(f.name for f in rejected_files)
            glc.logger.warning(
                f"The following files couldn't be parsed:\n\t{rejected_files}",
            )

        if not valid_files:
            msg = f"No valid file found in {folder}"
            raise FileNotFoundError(msg)

        return cls.from_files(
            files=valid_files,
            begin=begin,
            end=end,
            mode=mode,
            overlap=overlap,
            data_duration=data_duration,
            name=name,
            **kwargs,
        )

    @classmethod
    def _parse_file(
        cls: type[Self],
        file: Path,
        strptime_format: str,
        timezone: str | pytz.timezone | None,
        begin_timestamp: Timestamp,
        valid_files: list[TFile],
        rejected_files: list[Path],
    ) -> bool:
        if file.suffix.lower() not in cls.file_cls.supported_extensions:
            return False
        try:
            if strptime_format is None:
                f = cls.file_cls(file, begin=begin_timestamp, timezone=timezone)
            else:
                f = cls.file_cls(
                    file,
                    strptime_format=strptime_format,
                    timezone=timezone,
                )
            valid_files.append(f)
        except (ValueError, LibsndfileError):
            rejected_files.append(file)
            return False
        else:
            return True
