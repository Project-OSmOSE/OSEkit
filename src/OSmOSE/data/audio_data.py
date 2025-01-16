"""AudioData represent audio data scattered through different AudioFiles.

The AudioData has a collection of AudioItem.
The data is accessed via an AudioItem object per AudioFile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.data.audio_file import AudioFile
from OSmOSE.data.audio_item import AudioItem
from OSmOSE.data.base_data import BaseData
from OSmOSE.utils.audio_utils import resample

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timestamp


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

        """
        super().__init__(items=items, begin=begin, end=end)
        self._set_sample_rate(sample_rate=sample_rate)

    @property
    def nb_channels(self) -> int:
        """Number of channels of the audio data."""
        return max(
            [1] + [item.nb_channels for item in self.items if type(item) is AudioItem],
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the audio data."""
        data_length = round(self.sample_rate * self.duration.total_seconds())
        return data_length if self.nb_channels <= 1 else (data_length, self.nb_channels)

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)

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

    def get_value(self) -> np.ndarray:
        """Return the value of the audio data.

        The data from the audio file will be resampled if necessary.
        """
        data = np.empty(shape=self.shape)
        idx = 0
        for item in self.items:
            item_data = self._get_item_value(item)
            item_data = item_data[:min(item_data.shape[0], data.shape[0] - idx)]
            data[idx : idx + len(item_data)] = item_data
            idx += len(item_data)
        return data

    def write(self, folder: Path) -> None:
        """Write the audio data to file.

        Parameters
        ----------
        folder: pathlib.Path
            Folder in which to write the audio file.

        """
        super().write(path=folder)
        sf.write(folder / f"{self}.wav", self.get_value(), self.sample_rate)

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

    def divide(self, nb_subdata: int = 2) -> list[AudioData]:
        return [
            AudioData.from_base_data(base_data, self.sample_rate)
            for base_data in super().divide(nb_subdata)
        ]

    @classmethod
    def from_files(
        cls,
        files: list[AudioFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        sample_rate: float | None = None,
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

        Returns
        -------
        AudioData:
            The AudioData object.

        """
        return cls.from_base_data(BaseData.from_files(files, begin, end), sample_rate)

    @classmethod
    def from_base_data(
        cls,
        data: BaseData,
        sample_rate: float | None = None,
    ) -> AudioData:
        """Return an AudioData object from a BaseData object.

        Parameters
        ----------
        data: BaseData
            BaseData object to convert to AudioData.
        sample_rate: float | None
            Sample rate of the AudioData.

        Returns
        -------
        AudioData:
            The AudioData object.

        """
        return cls(
            items=[AudioItem.from_base_item(item) for item in data.items],
            sample_rate=sample_rate,
        )
