"""AudioData encapsulating to a collection of AudioItem objects."""

from __future__ import annotations

import numpy as np

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.data.audio_item import AudioItem
from OSmOSE.data.data_base import DataBase
from OSmOSE.utils.audio_utils import resample
import soundfile as sf
from pathlib import Path


class AudioData(DataBase):
    """AudioData encapsulating to a collection of AudioItem objects.

    The audio data can be retrieved from several Files through the Items.
    """

    item_cls = AudioItem

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
        self._check_sample_rates(sample_rate=sample_rate)

    @property
    def nb_channels(self) -> int:
        return max(
            [1] + [item.nb_channels for item in self.items if type(item) is AudioItem],
        )

    @property
    def shape(self):
        data_length = int(self.sample_rate * self.total_seconds)
        return data_length if self.nb_channels <= 1 else (data_length, self.nb_channels)

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)

    def _check_sample_rates(self, sample_rate: int | None = None) -> None:
        if sample_rate is not None or any(
            sample_rate := item.sample_rate
            for item in self.items
            if item.sample_rate is not None
        ):
            self.sample_rate = sample_rate
        else:
            self.sample_rate = None

    def get_value(self):
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
        item_data = item.get_value()
        if item.is_empty:
            return item_data.repeat(int(item.total_seconds * self.sample_rate))
        if item.sample_rate != self.sample_rate:
            return resample(item_data, item.sample_rate, self.sample_rate)
        return item_data
