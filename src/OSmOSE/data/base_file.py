"""BaseFile: Base class for the File objects.

A File object associates file-written data to timestamps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
    from pandas import Timestamp

from pathlib import Path

from OSmOSE.utils.timestamp_utils import strptime_from_text


class BaseFile:
    """Base class for the File objects.

    A File object associates file-written data to timestamps.
    """

    def __init__(
        self,
        path: PathLike | str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        strptime_format: str | None = None,
    ) -> None:
        """Initialize a File object with a path and timestamps.

        The begin timestamp can either be provided as a parameter or parsed from the filename according to the provided strptime_format.

        Parameters
        ----------
        path: PathLike | str
            Full path to the file.
        begin: pandas.Timestamp | None
            Timestamp corresponding to the first data point in the file.
            If it is not provided, strptime_format is mandatory.
            If both begin and strptime_format are provided, begin will overrule the timestamp embedded in the filename.
        end: pandas.Timestamp | None
            (Optional) Timestamp after the last data point in the file.
        strptime_format: str | None
            The strptime format used in the text.
            It should use valid strftime codes (https://strftime.org/).
            Example: '%y%m%d_%H:%M:%S'.

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
        self.end = end if end is not None else self.begin

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
