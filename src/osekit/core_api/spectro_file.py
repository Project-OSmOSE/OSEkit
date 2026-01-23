"""Spectro file associated with timestamps.

Spectro files are ``npz`` files with ``Time`` and ``Sxx`` arrays.
Metadata (``time_resolution``) are stored as separate arrays.
"""

from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import numpy as np
from pandas import Timedelta, Timestamp
from scipy.signal import ShortTimeFFT

from osekit.core_api.base_file import BaseFile

if TYPE_CHECKING:
    from os import PathLike

    import pytz


class SpectroFile(BaseFile):
    """Spectro file associated with timestamps.

    Spectro files are ``npz`` files with ``Time`` and ``Sxx`` arrays.
    Metadata (``time_resolution``) are stored as separate arrays.
    """

    supported_extensions: typing.ClassVar = [".npz"]

    def __init__(
        self,
        path: PathLike | str,
        begin: Timestamp | None = None,
        strptime_format: str | list[str] | None = None,
        timezone: str | pytz.timezone | None = None,
    ) -> None:
        """Initialize a ``SpectroFile`` object from a ``path`` and begin timestamp.

        The begin timestamp can either be provided as a parameter
        or parsed from the filename according to the provided ``strptime_format``.

        Parameters
        ----------
        path: PathLike | str
            Full path to the file.
        begin: pandas.Timestamp | None
            Timestamp corresponding to the first data bin in the file.
            If it is not provided, ``strptime_format`` is mandatory.
            If both ``begin`` and ``strptime_format`` are provided,
            ``begin`` will overrule the timestamp embedded in the filename.
        strptime_format: str | None
            The strptime format used in the text.
            It should use valid strftime codes (https://strftime.org/).
            Example: ``'%y%m%d_%H:%M:%S'``.
        timezone: str | pytz.timezone | None
            The timezone in which the file should be localized.
            If ``None``, the file begin/end will be tz-naive.
            If different from a timezone parsed from the filename, the timestamps'
            timezone will be converted from the parsed timezone
            to the specified timezone.

        """
        super().__init__(
            path=path,
            begin=begin,
            strptime_format=strptime_format,
            timezone=timezone,
        )
        self._read_metadata(path=path)

    def _read_metadata(self, path: PathLike) -> None:
        with np.load(path) as data:
            sample_rate = data["fs"][0]
            time = data["time"]
            freq = data["freq"]
            hop = int(data["hop"][0])
            window = data["window"]
            mfft = data["mfft"][0]
            timestamps = str(data["timestamps"])
            db_ref = data["db_ref"][0]
            v_lim = tuple(data["v_lim"])
            is_complex = np.iscomplexobj(data["sx"])

        self.sample_rate = sample_rate
        self.mfft = mfft

        self.begin, self.end = (Timestamp(t) for t in timestamps.split("_"))

        self.time = time
        self.time_resolution = (self.end - self.begin) / len(self.time)

        self.freq = freq

        self.window = window
        self.hop = hop

        self.sx_dtype = complex if is_complex else float

        self.db_ref = db_ref
        self.v_lim = v_lim

    def read(self, start: Timestamp, stop: Timestamp) -> np.ndarray:
        """Return the spectro data between ``start`` and ``stop`` from the file.

        The data is a 2D array representing the ``sxx`` values of the spectrogram.

        Parameters
        ----------
        start: pandas.Timestamp
            Timestamp corresponding to the first time bin to read.
        stop: pandas.Timestamp
            Timestamp after the last time bin to read.

        Returns
        -------
        numpy.ndarray:
            The spectrogram data between ``start`` and ``stop``.

        """
        with np.load(self.path) as data:
            time = data["time"]

            start_bin = (
                next(
                    (
                        idx
                        for idx, t in enumerate(time)
                        if self.begin + Timedelta(seconds=t) > start
                    ),
                    1,
                )
                - 1
            )
            start_bin = max(start_bin, 0)

            stop_bin = (
                next(
                    (
                        idx
                        for idx, t in list(enumerate(time))[::-1]
                        if self.begin + Timedelta(seconds=t) < stop
                    ),
                    len(time) - 2,
                )
                + 1
            )
            stop_bin = min(stop_bin, time.shape[0])

            return data["sx"][:, start_bin:stop_bin]

    def get_fft(self) -> ShortTimeFFT:
        """Return the ``ShortTimeFFT`` used for computing the spectrogram.

        Returns
        -------
        ShortTimeFFT:
            The ``ShortTimeFFT`` used for computing the spectrogram.
            It is instantiated back from the parameters stored in the ``npz`` file.

        """
        return ShortTimeFFT(
            win=self.window,
            hop=self.hop,
            fs=self.sample_rate,
            mfft=self.mfft,
        )
