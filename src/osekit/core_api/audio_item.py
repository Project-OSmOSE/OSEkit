"""``AudioItem`` corresponding to a portion of an ``AudioFile`` object."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from osekit.core_api.audio_file import AudioFile
from osekit.core_api.base_item import BaseItem

if TYPE_CHECKING:
    from pandas import Timestamp


class AudioItem(BaseItem[AudioFile]):
    """``AudioItem`` corresponding to a portion of an ``AudioFile`` object."""

    def __init__(
        self,
        file: AudioFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> None:
        """Initialize an ``AudioItem`` from an ``AudioFile`` and begin/end timestamps.

        Parameters
        ----------
        file: osekit.data.audio_file.AudioFile
            The ``AudioFile`` in which this Item belongs.
        begin: pandas.Timestamp (optional)
            The timestamp at which this item begins.
            It is defaulted to ``file.begin``.
        end: pandas.Timestamp (optional)
            The timestamp at which this item ends.
            It is defaulted to ``file.end``.

        """
        super().__init__(file=file, begin=begin, end=end)

    @property
    def sample_rate(self) -> float:
        """Sample rate of the associated ``AudioFile``."""
        return None if self.is_empty else self.file.sample_rate

    @property
    def nb_channels(self) -> int:
        """Number of channels in the associated ``AudioFile``."""
        return 1 if self.is_empty else self.file.channels

    @property
    def shape(self) -> tuple[int, int]:
        """Number of points in the audio item data."""
        if self.is_empty:
            return 0, self.nb_channels
        start_sample, end_sample = self.file.frames_indexes(self.begin, self.end)
        return end_sample - start_sample, self.nb_channels

    def get_value(self) -> np.ndarray:
        """Get the values from the File between the ``begin`` and ``stop`` timestamps.

        If the Item is empty, return a single ``0.`` per channel.
        """
        if self.is_empty:
            return np.zeros((1, self.nb_channels))
        return super().get_value()
