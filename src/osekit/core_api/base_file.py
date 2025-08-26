"""BaseFile: Base class for the File objects.

A File object associates file-written data to timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from osekit.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED,
    TIMESTAMP_FORMATS_EXPORTED_FILES,
)
from osekit.utils.timestamp_utils import localize_timestamp

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
    import pytz
    from pandas import Timestamp

from pathlib import Path

from pandas import Timedelta

from osekit.core_api.event import Event
from osekit.utils.timestamp_utils import strptime_from_text


class BaseFile(Event):
    """Base class for the File objects.

    A File object associates file-written data to timestamps.
    """

    def __init__(
        self,
        path: PathLike | str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        strptime_format: str | list[str] | None = None,
        timezone: str | pytz.timezone | None = None,
    ) -> None:
        """Initialize a File object with a path and timestamps.

        The begin timestamp can either be provided as a parameter
        or parsed from the filename according to the provided strptime_format.

        Parameters
        ----------
        path: PathLike | str
            Full path to the file.
        begin: pandas.Timestamp | None
            Timestamp corresponding to the first data point in the file.
            If it is not provided, strptime_format is mandatory.
            If both begin and strptime_format are provided,
            begin will overrule the timestamp embedded in the filename.
        end: pandas.Timestamp | None
            (Optional) Timestamp after the last data point in the file.
        strptime_format: str | None
            The strptime format used in the text.
            It should use valid strftime codes (https://strftime.org/).
            Example: '%y%m%d_%H:%M:%S'.
        timezone: str | pytz.timezone | None
            The timezone in which the file should be localized.
            If None, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.

        """
        self.path = Path(path)

        if begin is None and strptime_format is None:
            raise ValueError("Either begin or strptime_format must be specified")

        self.begin = (
            begin
            if begin is not None
            else strptime_from_text(
                text=self.path.name,
                datetime_template=strptime_format,
            )
        )

        if timezone:
            self.begin = localize_timestamp(self.begin, timezone)

        self.end = end if end is not None else (self.begin + Timedelta(seconds=1))

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray:
        """Return the data that is between start and stop from the file.

        This is an abstract method and should be overridden with actual implementations.

        Parameters
        ----------
        start: pandas.Timestamp
            Timestamp corresponding to the first data point to read.
        stop: pandas.Timestamp
            Timestamp after the last data point to read.

        Returns
        -------
        The data between start and stop.

        """

    def to_dict(self) -> dict:
        """Serialize a BaseFile to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the BaseFile.

        """
        return {
            "path": str(self.path),
            "begin": self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED),
            "end": self.end.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED),
        }

    @classmethod
    def from_dict(cls, serialized: dict) -> BaseFile:
        """Return a BaseFile object from a dictionary.

        Parameters
        ----------
        serialized: dict
            The serialized dictionary representing the BaseFile.

        Returns
        -------
        BaseFile:
            The deserialized BaseFile object.

        """
        path = serialized["path"]
        return cls(
            path=path,
            strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES,
        )

    def __hash__(self) -> int:
        """Overwrite hash magic method."""
        return hash(self.path)

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_LOCALIZED)

    def __eq__(self, other: BaseFile):
        """Override __eq__."""
        if not isinstance(other, BaseFile):
            return False
        return self.path == other.path and super().__eq__(other)

    def move(self, folder: Path) -> None:
        """Move the file to the target folder.

        Parameters
        ----------
        folder: Path
            destination folder where the file will be moved.

        """
        destination_path = folder / self.path.name
        folder.mkdir(exist_ok=True, parents=True)
        self.path.rename(destination_path)
        self.path = destination_path
