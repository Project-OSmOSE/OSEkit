"""AudioData represent audio data scattered through different AudioFiles.

The AudioData has a collection of AudioItem.
The data is accessed via an AudioItem object per AudioFile.
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf
from pandas import Timedelta, Timestamp

from osekit.config import (
    TIMESTAMP_FORMATS_EXPORTED_FILES,
)
from osekit.core_api.audio_file import AudioFile
from osekit.core_api.audio_item import AudioItem
from osekit.core_api.base_data import BaseData
from osekit.core_api.instrument import Instrument
from osekit.utils.audio_utils import Normalization, normalize, resample

if TYPE_CHECKING:
    from pathlib import Path


class AudioData(BaseData[AudioItem, AudioFile]):
    """AudioData represent audio data scattered through different AudioFiles.

    The AudioData has a collection of AudioItem.
    The data is accessed via an AudioItem object per AudioFile.
    """

    def __init__(
        self,
        items: list[AudioItem] | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        sample_rate: int | None = None,
        instrument: Instrument | None = None,
        normalization: Normalization = Normalization.RAW,
        normalization_values: dict | None = None,
    ) -> None:
        """Initialize an AudioData from a list of AudioItems.

        Parameters
        ----------
        items: list[AudioItem]
            List of the AudioItem constituting the AudioData.
        sample_rate: int
            The sample rate of the audio data.
        begin: Timestamp | None
            Only effective if items is None.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is None.
            Set the end of the empty data.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.

        """
        super().__init__(items=items, begin=begin, end=end)
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
    def shape(self) -> tuple[int, ...] | int:
        """Shape of the audio data."""
        data_length = round(self.sample_rate * self.duration.total_seconds())
        return data_length if self.nb_channels <= 1 else (data_length, self.nb_channels)

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

    def get_normalization_values(self) -> dict:
        values = self.get_raw_value()
        return {
            "mean": values.mean(),
            "peak": values.max(),
            "std": values.std(),
        }

    def __eq__(self, other: AudioData) -> bool:
        """Override __eq__."""
        return self.sample_rate == other.sample_rate and super().__eq__(other)

    def _set_sample_rate(self, sample_rate: int | None = None) -> None:
        """Set the AudioFile sample rate.

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
            If True, the AudioData will be bound to the written file.
            Its items will be replaced with a single item, which will match the whole
            new AudioFile.

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
        """Link the AudioData to an AudioFile in the folder.

        The given folder should contain a file named "str(self).wav".
        Linking is intended for AudioData objects that have already been written.
        After linking, the AudioData will have a single item with the same
        properties of the target AudioFile.

        Parameters
        ----------
        folder: Path
            Folder in which is located the AudioFile to which the AudioData instance
            should be linked.

        """
        file = AudioFile(
            path=folder / f"{self}.wav",
            strptime_format=TIMESTAMP_FORMATS_EXPORTED_FILES,
        )
        self.items = AudioData.from_files([file]).items

    def _get_item_value(self, item: AudioItem) -> np.ndarray:
        """Return the resampled (if needed) data from the audio item."""
        item_data = item.get_value()
        if item.is_empty:
            return item_data.repeat(
                round(item.duration.total_seconds() * self.sample_rate),
            )
        if item.sample_rate != self.sample_rate:
            return resample(item_data, item.sample_rate, self.sample_rate)
        return item_data

    def split(
        self,
        nb_subdata: int = 2,
        *,
        pass_normalization: bool = True,
    ) -> list[AudioData]:
        """Split the audio data object in the specified number of audio subdata.

        Parameters
        ----------
        nb_subdata: int
            Number of subdata in which to split the data.
        pass_normalization: bool
            If True, the normalization values (mean, std, peak) will be computed
            from the original audio data and passed to the split chunks.
            If the original AudioData is very long, this might lead to
            a RAM saturation.

        Returns
        -------
        list[AudioData]
            The list of AudioData subdata objects.

        """
        normalization_values = (
            None
            if not pass_normalization
            else self.normalization_values
            if any(self.normalization_values.values())
            else self.get_normalization_values()
        )
        return [
            AudioData.from_base_data(
                data=base_data,
                sample_rate=self.sample_rate,
                instrument=self.instrument,
                normalization=self.normalization,
                normalization_values=normalization_values,
            )
            for base_data in super().split(nb_subdata)
        ]

    def split_frames(
        self,
        start_frame: int = 0,
        stop_frame: int = -1,
        *,
        pass_normalization: bool = True,
    ) -> AudioData:
        """Return a new AudioData from a subpart of this AudioData's data.

        Parameters
        ----------
        start_frame: int
            First frame included in the new AudioData.
        stop_frame: int
            First frame after the last frame included in the new AudioData.
        pass_normalization: bool
            If True, the normalization values (mean, std, peak) will be computed
            from the original audio data and passed to the split chunks.
            If the original AudioData is very long, this might lead to
            a RAM saturation.

        Returns
        -------
        AudioData
            A new AudioData which data is included between start_frame and stop_frame.

        """
        if start_frame < 0:
            raise ValueError("Start_frame must be greater than or equal to 0.")
        if stop_frame < -1 or stop_frame > self.shape:
            raise ValueError("Stop_frame must be lower than the length of the data.")

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
        """Serialize an AudioData to a dictionary.

        Returns
        -------
        dict:
            The serialized dictionary representing the AudioData.

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
    def from_dict(cls, dictionary: dict) -> AudioData:
        """Deserialize an AudioData from a dictionary.

        Parameters
        ----------
        dictionary: dict
            The serialized dictionary representing the AudioData.

        Returns
        -------
        AudioData
            The deserialized AudioData.

        """
        base_data = BaseData.from_dict(dictionary)
        instrument = (
            None
            if dictionary["instrument"] is None
            else Instrument.from_dict(dictionary["instrument"])
        )
        return cls.from_base_data(
            data=base_data,
            sample_rate=dictionary["sample_rate"],
            normalization=Normalization(dictionary["normalization"]),
            normalization_values=dictionary["normalization_values"],
            instrument=instrument,
        )

    @classmethod
    def from_files(
        cls,
        files: list[AudioFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        sample_rate: float | None = None,
        instrument: Instrument | None = None,
        normalization: Normalization = Normalization.RAW,
        normalization_values: dict | None = None,
    ) -> AudioData:
        """Return an AudioData object from a list of AudioFiles.

        Parameters
        ----------
        files: list[AudioFile]
            List of AudioFiles containing the data.
        begin: Timestamp | None
            Begin of the data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the data object.
            Defaulted to the end of the last file.
        sample_rate: float | None
            Sample rate of the AudioData.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.
        normalization_values: dict|None
            Mean, peak and std values with which to normalize the data.

        Returns
        -------
        AudioData:
            The AudioData object.

        """
        return cls.from_base_data(
            data=BaseData.from_files(files, begin, end),
            sample_rate=sample_rate,
            instrument=instrument,
            normalization=normalization,
            normalization_values=normalization_values,
        )

    @classmethod
    def from_base_data(
        cls,
        data: BaseData,
        sample_rate: float | None = None,
        instrument: Instrument | None = None,
        normalization: Normalization = Normalization.RAW,
        normalization_values: dict | None = None,
    ) -> AudioData:
        """Return an AudioData object from a BaseData object.

        Parameters
        ----------
        data: BaseData
            BaseData object to convert to AudioData.
        sample_rate: float | None
            Sample rate of the AudioData.
        instrument: Instrument | None
            Instrument that might be used to obtain acoustic pressure from
            the wav audio data.
        normalization: Normalization
            The type of normalization to apply to the audio data.
        normalization_values: dict|None
            Mean, peak and std values with which to normalize the data.

        Returns
        -------
        AudioData:
            The AudioData object.

        """
        return cls(
            items=[AudioItem.from_base_item(item) for item in data.items],
            sample_rate=sample_rate,
            instrument=instrument,
            normalization=normalization,
            normalization_values=normalization_values,
        )
