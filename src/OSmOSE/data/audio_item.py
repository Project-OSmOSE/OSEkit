"""AudioItem corresponding to a portion of an AudioFile object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from OSmOSE.data.item_base import ItemBase

if TYPE_CHECKING:
    from pandas import Timestamp

    from OSmOSE.data.audio_file import AudioFile


class AudioItem(ItemBase):
    """AudioItem corresponding to a portion of an AudioFile object."""

    def __init__(
        self,
        file: AudioFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> None:
        """Initialize an AudioItem from an AudioItem and begin/end timestamps.

        Parameters
        ----------
        file: OSmOSE.data.audio_file.AudioFile
            The AudioFile in which this Item belongs.
        begin: pandas.Timestamp (optional)
            The timestamp at which this item begins.
            It is defaulted to the AudioFile begin.
        end: pandas.Timestamp (optional)
            The timestamp at which this item ends.
            It is defaulted to the AudioFile end.

        """
        super().__init__(file, begin, end)

    @property
    def sample_rate(self):
        return None if self.is_empty else self.file.metadata.samplerate

    @property
    def nb_channels(self):
        return 0 if self.is_empty else self.file.metadata.channels
