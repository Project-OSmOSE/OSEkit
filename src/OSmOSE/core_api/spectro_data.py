"""SpectroData represent spectrogram data retrieved from SpectroFiles.

The SpectroData has a collection of SpectroItem.
The data is accessed via a SpectroItem object per SpectroFile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import ShortTimeFFT

from OSmOSE.config import (
    TIMESTAMP_FORMAT_EXPORTED_FILES,
    TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
)
from OSmOSE.core_api.audio_data import AudioData
from OSmOSE.core_api.base_data import BaseData
from OSmOSE.core_api.spectro_file import SpectroFile
from OSmOSE.core_api.spectro_item import SpectroItem

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timestamp


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
        db_ref: float | None = None,
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
        self._sx_dtype = complex
        self._db_ref = db_ref

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
        nb_bytes_per_cell = 16 if self.sx_dtype is complex else 8
        return self.shape[0] * self.shape[1] * nb_bytes_per_cell

    @property
    def sx_dtype(self) -> type[complex]:
        """Data type used to represent the sx values. Should either be float or complex.

        If complex, the phase info will be included in the computed spectrum.
        If float, only the absolute value of the spectrum will be kept.

        """
        return self._sx_dtype

    @sx_dtype.setter
    def sx_dtype(self, dtype: type[complex]) -> [complex, float]:
        if dtype not in (complex, float):
            raise ValueError("dtype must be complex or float.")
        self._sx_dtype = dtype

    @property
    def db_ref(self) -> float:
        """Reference value for computing sx values in decibel.

        If no reference is specified (self._db_ref is None), the
        sx db values will be given in dB FS.
        """
        if self._db_ref is not None:
            return self._db_ref
        if (
            self.audio_data is not None
            and (instrument := self.audio_data.instrument) is not None
        ):
            return instrument.P_REF
        return 1.0

    @db_ref.setter
    def db_ref(self, db_ref: float) -> None:
        self._db_ref = db_ref

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ)

    def get_value(self) -> np.ndarray:
        """Return the Sx matrix of the spectrogram.

        The Sx matrix contains the absolute square of the STFT.
        """
        if not all(item.is_empty for item in self.items):
            return self._get_value_from_items(self.items)
        if not self.audio_data or not self.fft:
            raise ValueError("SpectroData should have either items or audio_data.")

        sx = self.fft.stft(
            self.audio_data.get_value_calibrated(reject_dc=True),
            padding="zeros",
        )

        if self.sx_dtype is float:
            sx = abs(sx) ** 2

        return sx

    def plot(self, ax: plt.Axes | None = None, sx: np.ndarray | None = None) -> None:
        """Plot the spectrogram on a specific Axes.

        Parameters
        ----------
        ax: plt.axes | None
            Axes on which the spectrogram should be plotted.
            Defaulted as the SpectroData.get_default_ax Axes.
        sx: np.ndarray | None
            Spectrogram sx values. Will be computed if not provided.

        """
        ax = ax if ax is not None else SpectroData.get_default_ax()
        sx = self.get_value() if sx is None else sx

        sx = self.to_db(sx)

        time = np.arange(sx.shape[1]) * self.duration.total_seconds() / sx.shape[1]
        freq = self.fft.f

        ax.pcolormesh(time, freq, sx, vmin=0, vmax=100)

    def to_db(self, sx: np.ndarray) -> np.ndarray:
        """Convert the sx values to dB.

        If the linked audio data has an Instrument parameter, the values are
        converted to dB SPL (re Instrument.P_REF).
        Otherwise, the values are converted to dB FS.

        Parameters
        ----------
        sx: np.ndarray
            Sx values of the spectrum.

        Returns
        -------
        np.ndarray
            Converted Sx values.

        """
        if self.sx_dtype is complex:
            sx = abs(sx) ** 2

        # sx has already been squared up, hence the 10*log for sx and 20*log for the ref
        return 10 * np.log10(sx + np.nextafter(0, 1)) - 20 * np.log10(self.db_ref)

    def save_spectrogram(
        self,
        folder: Path,
        ax: plt.Axes | None = None,
        sx: np.ndarray | None = None,
    ) -> None:
        """Export the spectrogram as a png image.

        Parameters
        ----------
        folder: Path
            Folder in which the spectrogram should be saved.
        ax: plt.Axes | None
            Axes on which the spectrogram should be plotted.
            Defaulted as the SpectroData.get_default_ax Axes.
        sx: np.ndarray | None
            Spectrogram sx values. Will be computed if not provided.

        """
        super().create_directories(path=folder)
        self.plot(ax, sx)
        plt.savefig(f"{folder / str(self)}", bbox_inches="tight", pad_inches=0)
        plt.close()

    def write(
        self,
        folder: Path,
        sx: np.ndarray | None = None,
        link: bool = False,
    ) -> None:
        """Write the Spectro data to file.

        Parameters
        ----------
        folder: pathlib.Path
            Folder in which to write the Spectro file.
        sx: np.ndarray | None
            Spectrogram sx values. Will be computed if not provided.
        link: bool
            If True, the SpectroData will be bound to the written npz file.
            Its items will be replaced with a single item, which will match the whole
            new SpectroFile.

        """
        super().create_directories(path=folder)
        sx = self.get_value() if sx is None else sx
        time = np.arange(sx.shape[1]) * self.duration.total_seconds() / sx.shape[1]
        freq = self.fft.f
        window = self.fft.win
        hop = [self.fft.hop]
        fs = [self.fft.fs]
        mfft = [self.fft.mfft]
        db_ref = [self.db_ref]
        timestamps = (str(t) for t in (self.begin, self.end))
        np.savez(
            file=folder / f"{self}.npz",
            fs=fs,
            time=time,
            freq=freq,
            window=window,
            hop=hop,
            sx=sx,
            mfft=mfft,
            db_ref=db_ref,
            timestamps="_".join(timestamps),
        )
        if link:
            self.link(folder=folder)

    def link(self, folder: Path) -> None:
        """Link the SpectroData to a SpectroFile in the folder.

        The given folder should contain a file named "str(self).npz".
        Linking is intended for SpectroData objects that have already been written to disk.
        After linking, the SpectroData will have a single item with the same
        properties of the target SpectroFile.

        Parameters
        ----------
        folder: Path
            Folder in which is located the SpectroFile to which the SpectroData instance should be linked.

        """
        file = SpectroFile(
            path=folder / f"{self}.npz",
            strptime_format=[
                TIMESTAMP_FORMAT_EXPORTED_FILES_WITH_TZ,
                TIMESTAMP_FORMAT_EXPORTED_FILES,
            ],
        )
        self.items = SpectroData.from_files([file]).items

    def link_audio_data(self, audio_data: AudioData) -> None:
        """Link the SpectroData to a given AudioData.

        Parameters
        ----------
        audio_data: AudioData
            The AudioData to which this SpectroData will be linked.

        """
        if self.begin != audio_data.begin:
            raise ValueError("The begin of the audio data doesn't match.")
        if self.end != audio_data.end:
            raise ValueError("The end of the audio data doesn't match.")
        if self.fft.fs != audio_data.sample_rate:
            raise ValueError("The sample rate of the audio data doesn't match.")
        self.audio_data = audio_data

    def split(self, nb_subdata: int = 2) -> list[SpectroData]:
        """Split the spectro data object in the specified number of spectro subdata.

        Parameters
        ----------
        nb_subdata: int
            Number of subdata in which to split the data.

        Returns
        -------
        list[SpectroData]
            The list of SpectroData subdata objects.

        """
        split_frames = list(
            np.linspace(0, self.audio_data.shape, nb_subdata + 1, dtype=int),
        )
        split_frames = [
            self.fft.nearest_k_p(frame) if idx < (len(split_frames) - 1) else frame
            for idx, frame in enumerate(split_frames)
        ]

        ad_split = [
            self.audio_data.split_frames(start_frame=a, stop_frame=b)
            for a, b in zip(split_frames, split_frames[1:])
        ]
        return [SpectroData.from_audio_data(ad, self.fft) for ad in ad_split]

    def _get_value_from_items(self, items: list[SpectroItem]) -> np.ndarray:
        if not all(
            np.array_equal(items[0].file.freq, i.file.freq)
            for i in items[1:]
            if not i.is_empty
        ):
            raise ValueError("Items don't have the same frequency bins.")

        if len({i.file.get_fft().delta_t for i in items if not i.is_empty}) > 1:
            raise ValueError("Items don't have the same time resolution.")

        output = items[0].get_value(fft=self.fft, sx_dtype=self.sx_dtype)
        for item in items[1:]:
            p1_le = self.fft.lower_border_end[1] - self.fft.p_min
            output = np.hstack(
                (
                    output[:, :-p1_le],
                    (
                        output[:, -p1_le:]
                        + item.get_value(fft=self.fft, sx_dtype=self.sx_dtype)[
                            :,
                            :p1_le,
                        ]
                    ),
                    item.get_value(fft=self.fft, sx_dtype=self.sx_dtype)[:, p1_le:],
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
        instance = cls.from_base_data(
            BaseData.from_files(files, begin, end),
            fft=files[0].get_fft(),
            db_ref=files[0].db_ref,
        )
        if not any(file.sx_dtype is complex for file in files):
            instance.sx_dtype = float
        return instance

    @classmethod
    def from_base_data(
        cls,
        data: BaseData,
        fft: ShortTimeFFT,
        db_ref: float | None = None,
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
        return cls(
            [SpectroItem.from_base_item(item) for item in data.items],
            fft=fft,
            db_ref=db_ref,
        )

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

    def to_dict(self, embed_sft: bool = True) -> dict:
        """Serialize a SpectroData to a dictionary.

        Parameters
        ----------
        embed_sft: bool
            If True, the SFT parameters will be included in the dictionary.
            In a case where multiple SpectroData that share a same SFT are serialized,
            SFT parameters shouldn't be included in the dictionary, as the window
            values might lead to large redundant data.
            Rather, the SFT parameters should be serialized in
            a SpectroDataset dictionary so that it can be only stored once
            for all SpectroData instances.

        Returns
        -------
        dict:
            The serialized dictionary representing the SpectroData.


        """
        base_dict = super().to_dict()
        audio_dict = {
            "audio_data": (
                None if self.audio_data is None else self.audio_data.to_dict()
            ),
        }
        sft_dict = {
            "sft": (
                {
                    "win": list(self.fft.win),
                    "hop": self.fft.hop,
                    "fs": self.fft.fs,
                    "mfft": self.fft.mfft,
                    "scale_to": self.fft.scaling,
                }
                if embed_sft
                else None
            ),
        }
        return base_dict | audio_dict | sft_dict

    @classmethod
    def from_dict(
        cls,
        dictionary: dict,
        sft: ShortTimeFFT | None = None,
    ) -> SpectroData:
        """Deserialize a SpectroData from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the AudioData.
        sft: ShortTimeFFT | None
            The ShortTimeFFT used to compute the spectrogram.
            If not provided, the SFT parameters must be included in the dictionary.

        Returns
        -------
        SpectroData
            The deserialized SpectroData.

        """
        if sft is None and dictionary["sft"] is None:
            raise ValueError("Missing sft")
        if sft is None:
            dictionary["sft"]["win"] = np.array(dictionary["sft"]["win"])
            sft = ShortTimeFFT(**dictionary["sft"])

        if dictionary["audio_data"] is None:
            base_data = BaseData.from_dict(dictionary)
            return cls.from_base_data(base_data, sft)

        audio_data = AudioData.from_dict(dictionary["audio_data"])
        return cls.from_audio_data(audio_data, sft)
