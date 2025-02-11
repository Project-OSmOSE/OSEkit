"""SpectroDataset is a collection of SpectroData objects.

SpectroDataset is a collection of SpectroData, with methods
that simplify repeated operations on the spectro data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from OSmOSE.data.base_dataset import BaseDataset
from OSmOSE.data.spectro_data import SpectroData
from OSmOSE.data.spectro_file import SpectroFile

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timedelta, Timestamp
    from scipy.signal import ShortTimeFFT

    from OSmOSE.data.audio_dataset import AudioDataset


class SpectroDataset(BaseDataset[SpectroData, SpectroFile]):
    """SpectroDataset is a collection of SpectroData objects.

    SpectroDataset is a collection of SpectroData, with methods
    that simplify repeated operations on the spectro data.

    """

    def __init__(self, data: list[SpectroData]) -> None:
        """Initialize a SpectroDataset."""
        super().__init__(data)

    @property
    def fft(self) -> ShortTimeFFT:
        """Return the fft of the spectro data."""
        return next(data.fft for data in self.data)

    @fft.setter
    def fft(self, fft: ShortTimeFFT) -> None:
        for data in self.data:
            data.fft = fft

    def save_spectrogram(self, folder: Path) -> None:
        """Export all spectrogram data as png images in the specified folder.

        Parameters
        ----------
        folder: Path
            Folder in which the spectrograms should be saved.

        """
        for data in self.data:
            data.save_spectrogram(folder)

    @classmethod
    def from_folder(
        cls,
        folder: Path,
        strptime_format: str,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        data_duration: Timedelta | None = None,
    ) -> SpectroDataset:
        """Return a SpectroDataset from a folder containing the spectro files.

        Parameters
        ----------
        folder: Path
            The folder containing the spectro files.
        strptime_format: str
            The strptime format of the timestamps in the spectro file names.
        begin: Timestamp | None
            The begin of the spectro dataset.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            The end of the spectro dataset.
            Defaulted to the end of the last file.
        data_duration: Timedelta | None
            Duration of the spectro data objects.
            If provided, spectro data will be evenly distributed between begin and end.
            Else, one data object will cover the whole time period.

        Returns
        -------
        Spectrodataset:
            The spectro dataset.

        """
        files = [
            SpectroFile(file, strptime_format=strptime_format)
            for file in folder.glob("*.npz")
        ]
        base_dataset = BaseDataset.from_files(files, begin, end, data_duration)
        return cls.from_base_dataset(base_dataset, files[0].get_fft())

    @classmethod
    def from_base_dataset(
        cls,
        base_dataset: BaseDataset,
        fft: ShortTimeFFT,
    ) -> SpectroDataset:
        """Return a SpectroDataset object from a BaseDataset object."""
        return cls(
            [SpectroData.from_base_data(data, fft) for data in base_dataset.data],
        )

    @classmethod
    def from_audio_dataset(
        cls,
        audio_dataset: AudioDataset,
        fft: ShortTimeFFT,
    ) -> SpectroDataset:
        """Return a SpectroDataset object from an AudioDataset object.

        The SpectroData is computed from the AudioData using the given fft.
        """
        return cls([SpectroData.from_audio_data(d, fft) for d in audio_dataset.data])
