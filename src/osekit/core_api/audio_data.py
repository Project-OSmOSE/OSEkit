"""AudioData represent audio data scattered through different AudioFiles.

The AudioData has a collection of AudioItem.
The data is accessed via an AudioItem object per AudioFile.
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING, Self

import numpy as np
import soundfile as sf
from pandas import Timedelta, Timestamp

from osekit.core_api.audio_file import AudioFile
from osekit.core_api.audio_item import AudioItem
from osekit.core_api.base_data import BaseData
from osekit.core_api.instrument import Instrument
from osekit.utils.audio_utils import Normalization, normalize, resample

if TYPE_CHECKING:
    from pathlib import Path


class AudioData(BaseData[AudioItem, AudioFile]):
    """``AudioData`` represent audio data scattered through different ``AudioFiles``.

    The ``AudioData`` has a collection of ``AudioItem``.
    The data is accessed via an ``AudioItem`` object per ``AudioFile``.
    """

    item_cls = AudioItem

    def __init__(
        self,
        items: list[AudioItem] | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        sample_rate: int | None = None,
        instrument: Instrument | None = None,
        normalization: Normalization = Normalization.RAW,
        normalization_values: dict | None = None,
    ) -> None:
        """Initialize an ``AudioData`` from a list of ``AudioItems``.

        Parameters
        ----------
        items: list[AudioItem]
            List of the ``AudioItem`` constituting the ``AudioData``.
        sample_rate: int
            The sample rate of the audio data.
        begin: Timestamp | None
            Only effective if items is None.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is None.
            Set the end of the empty data.
        name: str | None
            Name of the exported files.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.

        """
        super().__init__(items=items, begin=begin, end=end, name=name)
        self._set_sample_rate(sample_rate=sample_rate)
        self.instrument = instrument
        self.normalization = normalization
        self.normalization_values = normalization_values

    @property
    def nb_channels(self) -> int:
        """Number of channels of the audio data."""
        return max(
            [1] + [item.nb_channels for item in self.items if type(item) is AudioItem],
        )

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the audio data.

        First element is the number of data point in each channel,
        second element is the number of channels.

        """
        return self.length, self.nb_channels

    @property
    def length(self) -> int:
        """Number of data points in each channel."""
        return round(self.sample_rate * self.duration.total_seconds())

    @property
    def normalization(self) -> Normalization:
        """The type of normalization to apply to the audio data."""
        return self._normalization

    @normalization.setter
    def normalization(self, value: Normalization) -> None:
        self._normalization = value

    @property
    def normalization_values(self) -> dict:
        """Mean, peak and std values used for normalization."""
        return self._normalization_values

    @normalization_values.setter
    def normalization_values(self, value: dict | None) -> None:
        self._normalization_values = (
            value
            if value
            else {
                "mean": None,
                "peak": None,
                "std": None,
            }
        )

    @classmethod
    def _make_item(
        cls,
        file: AudioFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> AudioItem:
        """Make an ``AudioItem`` for a given ``AudioFile`` between begin and end timestamps.

        Parameters
        ----------
        file: AudioFile
            ``AudioFile`` of the item.
        begin: Timestamp
            Begin of the item.
        end:
            End of the item.

        Returns
        -------
        An AudioItem for the ``AudioFile`` file, between the begin and end timestamps.

        """
        return AudioItem(file=file, begin=begin, end=end)

    @classmethod
    def _make_file(cls, path: Path, begin: Timestamp) -> AudioFile:
        """Make an ``AudioFile`` from a path and a begin timestamp.

        Parameters
        ----------
        path: Path
            Path to the file.
        begin: Timestamp
            Begin of the file.

        Returns
        -------
        AudioFile:
        The ``AudioFile`` instance.

        """
        return AudioFile(path=path, begin=begin)

    def get_normalization_values(self) -> dict:
        """Return the values used for normalizing the audio data.

        Returns
        -------
        dict:
            "mean": mean value to substract to center values on 0.
            "peak": peak value for PEAK normalization
            "std": standard deviation used for z-score normalization

        """
        values = np.array(self.get_raw_value())
        return {
            "mean": values.mean(),
            "peak": values.max(),
            "std": values.std(),
        }

    def __eq__(self, other: AudioData) -> bool:
        """Override __eq__."""
        return self.sample_rate == other.sample_rate and super().__eq__(other)

    def _set_sample_rate(self, sample_rate: int | None = None) -> None:
        """Set the ``AudioFile`` sample rate.

        If the sample_rate is specified, it is set.
        If it is not specified, it is set to the sampling rate of the
        first item that has one.
        Else, it is set to None.
        """
        if sample_rate is not None:
            self.sample_rate = sample_rate
            return
        if sr := next(
            (item.sample_rate for item in self.items if item.sample_rate is not None),
            None,
        ):
            self.sample_rate = sr
            return
        self.sample_rate = None

    def get_raw_value(self) -> np.ndarray:
        """Return the raw value of the audio data before normalization.

        The data from the audio file will be resampled if necessary.

        Returns
        -------
        np.ndarray:
            The value of the audio data.

        """
        data = np.empty(shape=self.shape)
        idx = 0
        for item in self.items:
            item_data = self._get_item_value(item)
            item_data = item_data[: min(item_data.shape[0], data.shape[0] - idx)]
            data[idx : idx + len(item_data)] = item_data
            idx += len(item_data)
        return data

    def get_value(self) -> np.ndarray:
        """Return the value of the audio data.

        The data from the audio file will be resampled if necessary.

        Returns
        -------
        np.ndarray:
            The value of the audio data.

        """
        return normalize(
            values=self.get_raw_value(),
            normalization=self.normalization,
            **self.normalization_values,
        )

    def get_value_calibrated(self) -> np.ndarray:
        """Return the value of the audio data accounting for the calibration factor.

        If the instrument parameter of the audio data is not None, the returned value is
        calibrated in units of Pa.

        Returns
        -------
        np.ndarray:
            The calibrated value of the audio data.

        """
        raw_data = self.get_value()
        calibration_factor = (
            1.0 if self.instrument is None else self.instrument.end_to_end
        )
        return raw_data * calibration_factor

    def write(
        self,
        folder: Path,
        subtype: str | None = None,
        *,
        link: bool = False,
    ) -> None:
        """Write the audio data to file.

        Parameters
        ----------
        folder: pathlib.Path
            Folder in which to write the audio file.
        subtype: str | None
            Subtype as provided by the soundfile module.
            Defaulted as the default 16-bit PCM for WAV audio files.
        link: bool
            If True, the ``AudioData`` will be bound to the written file.
            Its items will be replaced with a single item, which will match the whole
            new ``AudioFile``.

        """
        super().create_directories(path=folder)
        sf.write(
            folder / f"{self}.wav",
            self.get_value(),
            self.sample_rate,
            subtype=subtype,
        )
        if link:
            self.link(folder=folder)

    def link(self, folder: Path) -> None:
        """Link the ``AudioData`` to an ``AudioFile`` in the folder.

        The given folder should contain a file named ``"str(self).wav"``.
        Linking is intended for ``AudioData`` objects that have already been written.
        After linking, the ``AudioData`` will have a single item with the same
        properties of the target ``AudioFile``.

        Parameters
        ----------
        folder: Path
            Folder in which is located the ``AudioFile`` to which the ``AudioData`` instance
            should be linked.

        """
        file = AudioFile(
            path=folder / f"{self}.wav",
            begin=self.begin,
        )
        self.items = AudioData.from_files([file]).items

    def _get_item_value(self, item: AudioItem) -> np.ndarray:
        """Return the resampled (if needed) data from the audio item."""
        item_data = item.get_value()
        if item.is_empty:
            return item_data.repeat(
                round(item.duration.total_seconds() * self.sample_rate),
                axis=0,
            )
        if item.sample_rate != self.sample_rate:
            return resample(item_data, item.sample_rate, self.sample_rate)
        return item_data

    def split(
        self,
        nb_subdata: int = 2,
        *,
        pass_normalization: bool = True,
    ) -> list[Self]:
        """Split the audio data object in the specified number of audio subdata.

        Parameters
        ----------
        nb_subdata: int
            Number of subdata in which to split the data.
        pass_normalization: bool
            If True, the normalization values (mean, std, peak) will be computed
            from the original audio data and passed to the split chunks.
            If the original ``AudioData`` is very long, this might lead to
            a RAM saturation.

        Returns
        -------
        list[AudioData]
            The list of ``AudioData`` subdata objects.

        """
        normalization_values = (
            None
            if not pass_normalization
            else self.normalization_values
            if any(self.normalization_values.values())
            else self.get_normalization_values()
        )
        return super().split(
            nb_subdata=nb_subdata,
            normalization_values=normalization_values,
        )

    def _make_split_data(
        self,
        files: list[AudioFile],
        begin: Timestamp,
        end: Timestamp,
        **kwargs: tuple[float, float, float],
    ) -> AudioData:
        """Return an ``AudioData`` object after an ``AudioData.split()`` call.

        Parameters
        ----------
        files: list[AudioFile]
            The ``AudioFiles`` of the original ``AudioData``.
        begin: Timestamp
            The begin timestamp of the split ``AudioData``.
        end: Timestamp
            The end timestamp of the split ``AudioData``.
        kwargs:
            normalization_values: tuple[float, float, float]
                Values used for normalizing the split ``AudioData``.

        Returns
        -------
        AudioData:
            The ``AudioData`` instance.

        """
        return AudioData.from_files(
            files=files,
            begin=begin,
            end=end,
            sample_rate=self.sample_rate,
            instrument=self.instrument,
            normalization=self.normalization,
            normalization_values=kwargs["normalization_values"],
        )

    def split_frames(
        self,
        start_frame: int = 0,
        stop_frame: int = -1,
        *,
        pass_normalization: bool = True,
    ) -> AudioData:
        """Return a new ``AudioData`` from a subpart of this ``AudioData``'s data.

        Parameters
        ----------
        start_frame: int
            First frame included in the new ``AudioData``.
        stop_frame: int
            First frame after the last frame included in the new ``AudioData``.
        pass_normalization: bool
            If ``True``, the normalization values (mean, std, peak) will be computed
            from the original audio data and passed to the split chunks.
            If the original ``AudioData`` is very long, this might lead to
            a RAM saturation.

        Returns
        -------
        AudioData
            A new ``AudioData`` which data is included between start_frame and stop_frame.

        """
        if start_frame < 0:
            msg = "Start_frame must be greater than or equal to 0."
            raise ValueError(msg)
        if stop_frame < -1 or stop_frame > self.length:
            msg = "Stop_frame must be lower than the length of the data."
            raise ValueError(msg)

        start_timestamp = self.begin + Timedelta(
            seconds=ceil(start_frame / self.sample_rate * 1e9) / 1e9,
        )
        stop_timestamp = (
            self.end
            if stop_frame == -1
            else self.begin + Timedelta(seconds=stop_frame / self.sample_rate)
        )
        normalization_values = (
            None
            if not pass_normalization
            else self.normalization_values
            if any(self.normalization_values.values())
            else self.get_normalization_values()
        )
        return AudioData.from_files(
            list(self.files),
            start_timestamp,
            stop_timestamp,
            sample_rate=self.sample_rate,
            instrument=self.instrument,
            normalization=self.normalization,
            normalization_values=normalization_values,
        )

    def to_dict(self) -> dict:
        """Serialize an ``AudioData`` to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the ``AudioData``.

        """
        base_dict = super().to_dict()
        instrument_dict = {
            "instrument": (
                None if self.instrument is None else self.instrument.to_dict()
            ),
        }
        return (
            base_dict
            | instrument_dict
            | {
                "sample_rate": self.sample_rate,
                "normalization": self.normalization.value,
                "normalization_values": self.normalization_values,
            }
        )

    @classmethod
    def _from_base_dict(
        cls,
        dictionary: dict,
        files: list[AudioFile],
        begin: Timestamp,
        end: Timestamp,
        **kwargs,  # noqa: ANN003
    ) -> AudioData:
        """Deserialize the ``AudioData``-specific parts of a Data dictionary.

        This method is called within the ``BaseData.from_dict()`` method, which
        deserializes the base files, begin and end parameters.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the ``AudioData``.
        files: list[AudioFile]
            The list of deserialized ``AudioFiles``.
        begin: Timestamp
            The deserialized begin timestamp.
        end: Timestamp
            The deserialized end timestamp.
        kwargs:
            None.

        Returns
        -------
        AudioData
            The deserialized ``AudioData``.

        """
        instrument = (
            None
            if dictionary["instrument"] is None
            else Instrument.from_dict(dictionary["instrument"])
        )
        return cls.from_files(
            files=files,
            begin=begin,
            end=end,
            instrument=instrument,
            sample_rate=dictionary["sample_rate"],
            normalization=Normalization(dictionary["normalization"]),
            normalization_values=dictionary["normalization_values"],
        )

    @classmethod
    def from_files(
        cls,
        files: list[AudioFile],  # The method is redefined just to specify the type
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        name: str | None = None,
        **kwargs,  # noqa: ANN003
    ) -> AudioData:
        """Return an ``AudioData`` object from a list of ``AudioFiles``.

        Parameters
        ----------
        files: list[AudioFile]
            List of ``AudioFiles`` containing the data.
        begin: Timestamp | None
            Begin of the data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the data object.
            Defaulted to the end of the last file.
        name: str | None
            Name of the exported files.
        kwargs
            Keyword arguments that are passed to the cls constructor.

            sample_rate: int
            The sample rate of the audio data.

            instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.

            normalization: Normalization
            The type of normalization to apply to the audio data.

        Returns
        -------
        Self:
        The ``AudioData`` object.

        """
        return super().from_files(
            files=files,  # This way, this static error doesn't appear to the user
            begin=begin,
            end=end,
            name=name,
            **kwargs,
        )
