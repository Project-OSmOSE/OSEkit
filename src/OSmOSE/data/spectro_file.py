"""Spectro file associated with timestamps.

Spectro files are npz files with Time and Sxx arrays.
Metadata (time_resolution) are stored as separate arrays.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas import Timedelta, Timestamp

from OSmOSE.data.base_file import BaseFile

if TYPE_CHECKING:
    from os import PathLike


class SpectroFile(BaseFile):
    """Spectro file associated with timestamps.

    Spectro files are npz files with Time and Sxx arrays.
    Metadata (time_resolution) are stored as separate arrays.
    """

    def __init__(
        self,
        path: PathLike | str,
        begin: Timestamp | None = None,
        strptime_format: str | None = None,
    ) -> None:
        """Initialize a SpectroFile object with a path and a begin timestamp.

        The begin timestamp can either be provided as a parameter
         or parsed from the filename according to the provided strptime_format.

        Parameters
        ----------
        path: PathLike | str
            Full path to the file.
        begin: pandas.Timestamp | None
            Timestamp corresponding to the first data bin in the file.
            If it is not provided, strptime_format is mandatory.
            If both begin and strptime_format are provided,
            begin will overrule the timestamp embedded in the filename.
        strptime_format: str | None
            The strptime format used in the text.
            It should use valid strftime codes (https://strftime.org/).
            Example: '%y%m%d_%H:%M:%S'.

        """
        super().__init__(path=path, begin=begin, strptime_format=strptime_format)
        self._read_metadata(path=path)
        self.end = self.begin + self.duration

    def _read_metadata(self, path: PathLike) -> None:
        with np.load(path) as data:
            sample_rate = data["fs"][0]
            time = data["time"]
            freq = data["freq"]

        self.sample_rate = sample_rate

        delta_times = [Timedelta(seconds=time[i] - time[i-1]).round(freq = "ns") for i in range(1,time.shape[0])]
        most_frequent_delta_time = max(((v, delta_times.count(v)) for v in set(delta_times)), key=lambda i: i[1])[0]
        self.time_resolution = most_frequent_delta_time
        self.end = (self.begin + Timedelta(seconds = time[-1]) + self.time_resolution).round(freq = "us")

        self.freq = freq

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray:
        """Return the spectro data between start and stop from the file.

        The data is a 2D array representing the sxx values of the spectrogram.

        Parameters
        ----------
        start: pandas.Timestamp
            Timestamp corresponding to the first time bin to read.
        stop: pandas.Timestamp
            Timestamp after the last time bin to read.

        Returns
        -------
        numpy.ndarray:
            The spectrogram data between start and stop.

        """
        start_bin = round((start - self.begin) / self.time_resolution)
        stop_bin = round((stop - self.begin) / self.time_resolution)
        with np.load(self.path) as data:
            return data["Sxx"][:, start_bin:stop_bin]

    @classmethod
    def from_base_file(cls, file: BaseFile) -> SpectroFile:
        """Return a SpectroFile object from a BaseFile object."""
        return cls(path=file.path, begin=file.begin)
