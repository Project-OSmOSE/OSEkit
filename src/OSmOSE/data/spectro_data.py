"""SpectroData represent spectrogram data retrieved from SpectroFiles.

The SpectroData has a collection of SpectroItem.
The data is accessed via a SpectroItem object per SpectroFile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.data.base_data import BaseData
from OSmOSE.data.spectro_file import SpectroFile
from OSmOSE.data.spectro_item import SpectroItem

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timestamp
    from scipy.signal import ShortTimeFFT

    from OSmOSE.data.audio_data import AudioData


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
        fft: ShortTimeFFT | None = None,
    ) -> None:
        """Initialize a SpectroData from a list of SpectroItems.

        Parameters
        ----------
        items: list[SpectroItem]
            List of the SpectroItem constituting the SpectroData.
        audio_data: AudioData
            The audio data from which to compute the spectrogram.
        begin: Timestamp | None
            Only effective if items is None.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is None.
            Set the end of the empty data.
        fft: ShortTimeFFT
            The short time FFT used for computing the spectrogram.

        """
        super().__init__(items=items, begin=begin, end=end)
        self.audio_data = audio_data
        self.fft = fft

    @staticmethod
    def get_default_ax() -> plt.Axes:
        """Return a default-formatted Axes on a new figure.

        The default OSmOSE spectrograms are plotted on wide, borderless spectrograms.
        This method set the default figure and axes parameters.

        Returns
        -------
        plt.Axes:
            The default Axes on a new figure.

        """
        # Legacy OSEkit behaviour.
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
            top=1,
            bottom=0,
            right=1,
            left=0,
            hspace=0,
            wspace=0,
        )
        return ax

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the Spectro data."""
        return self.fft.f_pts, self.fft.p_num(
            int(self.fft.fs * self.duration.total_seconds()),
        )

    @property
    def nb_bytes(self) -> int:
        """Total bytes consumed by the spectro values."""
        return self.shape[0] * self.shape[1] * 8

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)

    def get_value(self) -> np.ndarray:
        """Return the Sx matrix of the spectrogram.

        The Sx matrix contains the absolute square of the STFT.
        """
        if not all(item.is_empty for item in self.items):
            return self._get_value_from_items(self.items)
        if not self.audio_data or not self.fft:
            raise ValueError("SpectroData should have either items or audio_data.")

        return self.fft.spectrogram(self.audio_data.get_value(), padding="even")

    def plot(self, ax: plt.Axes | None = None) -> None:
        """Plot the spectrogram on a specific Axes.

        Parameters
        ----------
        ax: plt.axes | None
            Axes on which the spectrogram should be plotted.
            Defaulted as the SpectroData.get_default_ax Axes.

        """
        ax = ax if ax is not None else SpectroData.get_default_ax()
        sx = self.get_value()
        sx = 10 * np.log10(abs(sx) + np.nextafter(0, 1))
        time = np.arange(sx.shape[1]) * self.duration.total_seconds() / sx.shape[1]
        freq = self.fft.f
        ax.pcolormesh(time, freq, sx, vmin=-120, vmax=0)

    def save_spectrogram(self, folder: Path, ax: plt.Axes | None = None) -> None:
        """Export the spectrogram as a png image.

        Parameters
        ----------
        folder: Path
            Folder in which the spectrogram should be saved.
        ax: plt.Axes | None
            Axes on which the spectrogram should be plotted.
            Defaulted as the SpectroData.get_default_ax Axes.

        """
        super().write(folder)
        self.plot(ax)
        plt.savefig(f"{folder / str(self)}", bbox_inches="tight", pad_inches=0)
        plt.close()

    def write(self, folder: Path) -> None:
        """Write the Spectro data to file.

        Parameters
        ----------
        folder: pathlib.Path
            Folder in which to write the Spectro file.

        """
        super().create_directories(path=folder)
        sx = self.get_value()
        time = np.arange(sx.shape[1]) * self.duration.total_seconds() / sx.shape[1]
        freq = self.fft.f
        window = self.fft.win
        hop = [self.fft.hop]
        fs = [self.fft.fs]
        mfft = [self.fft.mfft]
        np.savez(
            file=folder / f"{self}.npz",
            fs=fs,
            time=time,
            freq=freq,
            window=window,
            hop=hop,
            sx=sx,
            mfft=mfft,
        )

    def _get_value_from_items(self, items: list[SpectroItem]) -> np.ndarray:
        if not all(
            np.array_equal(items[0].file.freq, i.file.freq)
            for i in items[1:]
            if not i.is_empty
        ):
            raise ValueError("Items don't have the same frequency bins.")

        if len({i.file.get_fft().delta_t for i in items if not i.is_empty}) > 1:
            raise ValueError("Items don't have the same time resolution.")

        output = items[0].get_value(fft=self.fft)
        for item in items[1:]:
            p1_le = self.fft.lower_border_end[1] - self.fft.p_min - 1
            output = np.hstack(
                (
                    output[:, :-p1_le],
                    (output[:, -p1_le:] + item.get_value(fft=self.fft)[:, :p1_le]) / 2,
                    item.get_value(fft=self.fft)[:, p1_le:],
                ),
            )
        return output

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
        return cls.from_base_data(
            BaseData.from_files(files, begin, end),
            fft=files[0].get_fft(),
        )

    @classmethod
    def from_base_data(
        cls,
        data: BaseData,
        fft: ShortTimeFFT,
    ) -> SpectroData:
        """Return an SpectroData object from a BaseData object.

        Parameters
        ----------
        data: BaseData
            BaseData object to convert to SpectroData.
        fft: ShortTimeFFT
            The ShortTimeFFT used to compute the spectrogram.

        Returns
        -------
        SpectroData:
            The SpectroData object.

        """
        return cls([SpectroItem.from_base_item(item) for item in data.items], fft=fft)

    @classmethod
    def from_audio_data(cls, data: AudioData, fft: ShortTimeFFT) -> SpectroData:
        """Instantiate a SpectroData object from a AudioData object.

        Parameters
        ----------
        data: AudioData
            Audio data from which the SpectroData should be computed.
        fft: ShortTimeFFT
            The ShortTimeFFT used to compute the spectrogram.

        Returns
        -------
        SpectroData:
            The SpectroData object.

        """
        return cls(audio_data=data, fft=fft, begin=data.begin, end=data.end)
