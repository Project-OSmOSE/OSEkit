"""AudioData encapsulating to a collection of AudioItem objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.data.audio_file import AudioFile
from OSmOSE.data.audio_item import AudioItem
from OSmOSE.data.data_base import DataBase
from OSmOSE.utils.audio_utils import resample

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timestamp


class AudioData(DataBase[AudioItem, AudioFile]):
    """AudioData encapsulating to a collection of AudioItem objects.

    The audio data can be retrieved from several Files through the Items.
    """

    def __init__(self, items: list[AudioItem], sample_rate: int | None = None) -> None:
        """Initialize an AudioData from a list of AudioItems.

        Parameters
        ----------
        items: list[AudioItem]
            List of the AudioItem constituting the AudioData.
        sample_rate: int
            The sample rate of the audio data.

        """
        super().__init__(items)
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
        data_length = int(self.sample_rate * self.total_seconds)
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
        if sample_rate is not None or any(
            sample_rate := item.sample_rate
            for item in self.items
            if item.sample_rate is not None
        ):
            self.sample_rate = sample_rate
        else:
            self.sample_rate = None

    def get_value(self) -> np.ndarray:
        """Return the value of the audio data.

        The data from the audio file will be resampled if necessary.
        """
        data = np.empty(shape=self.shape)
        idx = 0
        for item in self.items:
            item_data = self._get_item_value(item)
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
        sf.write(folder / f"{self}.wav", self.get_value(), self.sample_rate)

    def _get_item_value(self, item: AudioItem) -> np.ndarray:
        """Return the resampled (if needed) data from the audio item."""
        item_data = item.get_value()
        if item.is_empty:
            return item_data.repeat(int(item.total_seconds * self.sample_rate))
        if item.sample_rate != self.sample_rate:
            return resample(item_data, item.sample_rate, self.sample_rate)
        return item_data

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

        Returns
        -------
        DataBase[AudioItem, AudioFile]:
        The AudioData object.

        """
        return cls.from_base_data(DataBase.from_files(files, begin, end), sample_rate)
        items_base = DataBase.items_from_files(files, begin, end)
        audio_items = [
            AudioItem(file=item.file, begin=item.begin, end=item.end)
            for item in items_base
        ]
        return cls(audio_items)

    @classmethod
    def from_base_data(
        cls, data: DataBase, sample_rate: float | None = None
    ) -> AudioData:
        return cls([AudioItem.from_base_item(item) for item in data.items], sample_rate)
