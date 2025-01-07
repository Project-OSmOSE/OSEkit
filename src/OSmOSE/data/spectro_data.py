"""SpectroData represent spectrogram data retrieved from SpectroFiles.

The SpectroData has a collection of SpectroItem.
The data is accessed via a SpectroItem object per SpectroFile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.data.audio_data import AudioData
from OSmOSE.data.base_data import BaseData
from OSmOSE.data.spectro_file import SpectroFile
from OSmOSE.data.spectro_item import SpectroItem

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timedelta, Timestamp


class SpectroData(BaseData[SpectroItem, SpectroFile]):
    """SpectroData represent Spectro data scattered through different SpectroFiles.

    The SpectroData has a collection of SpectroItem.
    The data is accessed via a SpectroItem object per SpectroFile.
    """

    def __init__(
        self,
        items: list[SpectroItem] | None = None,
        audio_data: AudioData = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        time_resolution: Timedelta | None = None,
        fft : ShortTimeFFT | None = None,
    ) -> None:
        """Initialize a SpectroData from a list of SpectroItems.

        Parameters
        ----------
        items: list[SpectroItem]
            List of the SpectroItem constituting the SpectroData.
        time_resolution: Timedelta
            The time resolution of the Spectro data.
        begin: Timestamp | None
            Only effective if items is None.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is None.
            Set the end of the empty data.

        """
        super().__init__(items=items, begin=begin, end=end)
        # self._set_time_resolution(time_resolution=time_resolution)
        self.audio_data = audio_data
        self.fft = fft
        self._ax = None

    @property
    def ax(self) -> plt.Axes:
        if self._ax is not None:
            return self._ax

        # Legacy OSEkit behaviour, done in the getter so that plt figure is created on demand only.
        _, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(1.3 * 1800 / 100, 1.3 * 512 / 100),
            dpi=100,
        )

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.axis("off")
        plt.subplots_adjust(
            top=1, bottom=0, right=1, left=0, hspace=0, wspace=0,
        )
        self.ax = ax
        return ax

    @ax.setter
    def ax(self, ax: plt.Axes | None) -> None:
        self._ax = ax

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the Spectro data."""
        return max(item.shape[0] for item in self.items), sum(
            item.shape[1] for item in self.items
        )

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)

    def _set_time_resolution(self, time_resolution: Timedelta) -> None:
        """Set the SpectroFile time resolution."""
        if len(tr := {item.time_resolution for item in self.items}) > 1:
            raise ValueError("Items don't have the same time resolution")
        self.time_resolution = tr.pop() if len(tr) == 1 else time_resolution

    def get_value(self) -> np.ndarray:
        """Return the value of the Spectro data.

        The data from the Spectro file will be resampled if necessary.
        """
        if not all(item.is_empty for item in self.items):
            return self._get_value_from_items(self.items)
        if not self.audio_data or not self.fft:
            raise ValueError("SpectroData has not been initialized")

        return self.fft.spectrogram(self.audio_data.get_value())

    def plot(self):
        sx = self.get_value()
        time = np.arange(sx.shape[1]) * self.duration.total_seconds() / sx.shape[1]
        freq = self.fft.f
        log_spectro = 10 * np.log10(abs(sx) + 1e-12)
        self.ax.pcolormesh(time, freq, log_spectro)

    def save_spectrogram(self, folder: Path) -> None:
        super().write(folder)
        self.plot()
        plt.figure(self.ax.get_figure().number)
        plt.savefig(f"{folder / str(self)}", bbox_inches="tight", pad_inches=0)
        plt.close()

    def write(self, folder: Path) -> None:
        """Write the Spectro data to file.

        Parameters
        ----------
        folder: pathlib.Path
            Folder in which to write the Spectro file.

        """
        super().write(path=folder)
        # TODO: implement npz write

    def _get_value_from_items(self, items: list[SpectroItem]) -> np.ndarray:
        data = np.zeros(shape=self.shape)
        idx = 0
        for item in self.items:
            item_data = self._get_item_value(item)
            time_bins = item_data.shape[1]
            data[:, idx: idx + time_bins] = item_data
            idx += time_bins
        return data

    def _get_item_value(self, item: SpectroItem) -> np.ndarray:
        """Return the resampled (if needed) data from the Spectro item."""
        item_data = item.get_value()
        if item.is_empty:
            return item_data.repeat(round(item.duration / self.time_resolution))
        if item.time_resolution != self.time_resolution:
            raise ValueError("Time resolutions don't match.")
        return item_data

    @classmethod
    def from_files(
        cls,
        files: list[SpectroFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> SpectroData:
        """Return a SpectroData object from a list of SpectroFiles.

        Parameters
        ----------
        files: list[SpectroFile]
            List of SpectroFiles containing the data.
        begin: Timestamp | None
            Begin of the data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the data object.
            Defaulted to the end of the last file.

        Returns
        -------
        SpectroData:
            The SpectroData object.

        """
        return cls.from_base_data(BaseData.from_files(files, begin, end))

    @classmethod
    def from_base_data(
        cls,
        data: BaseData,
    ) -> SpectroData:
        """Return an SpectroData object from a BaseData object.

        Parameters
        ----------
        data: BaseData
            BaseData object to convert to SpectroData.

        Returns
        -------
        SpectroData:
            The SpectroData object.

        """
        return cls([SpectroItem.from_base_item(item) for item in data.items])

    @classmethod
    def from_audio_data(cls, data: AudioData, fft: ShortTimeFFT) -> SpectroData:
        return cls(audio_data=data, fft=fft, begin=data.begin, end=data.end)
