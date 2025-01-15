"""Audio file associated with timestamps."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike

    import numpy as np
from pandas import Timedelta, Timestamp

from OSmOSE.data import audio_file_manager as afm
from OSmOSE.data.base_file import BaseFile


class AudioFile(BaseFile):
    """Audio file associated with timestamps."""

    def __init__(
        self,
        path: PathLike | str,
        begin: Timestamp | None = None,
        strptime_format: str | None = None,
    ) -> None:
        """Initialize an AudioFile object with a path and a begin timestamp.

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
        strptime_format: str | None
            The strptime format used in the text.
            It should use valid strftime codes (https://strftime.org/).
            Example: '%y%m%d_%H:%M:%S'.

        """
        super().__init__(path=path, begin=begin, strptime_format=strptime_format)
        sample_rate, frames, channels = afm.info(path)
        duration = frames / sample_rate
        self.sample_rate = sample_rate
        self.channels = channels
        self.end = self.begin + Timedelta(seconds=duration)

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray:
        """Return the audio data between start and stop from the file.

        Parameters
        ----------
        start: pandas.Timestamp
            Timestamp corresponding to the first data point to read.
        stop: pandas.Timestamp
            Timestamp after the last data point to read.

        Returns
        -------
        numpy.ndarray:
            The audio data between start and stop.

        """
        sample_rate = self.sample_rate
        start_sample = round((start - self.begin).total_seconds() * sample_rate)
        stop_sample = round((stop - self.begin).total_seconds() * sample_rate)
        return afm.read(self.path, start=start_sample, stop=stop_sample)

    @classmethod
    def from_base_file(cls, file: BaseFile) -> AudioFile:
        """Return an AudioFile object from a BaseFile object."""
        return cls(path=file.path, begin=file.begin)
