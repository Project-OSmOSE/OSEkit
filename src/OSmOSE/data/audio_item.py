"""AudioItem corresponding to a portion of an AudioFile object."""

from __future__ import annotations

from typing import TYPE_CHECKING

from OSmOSE.data.audio_file import AudioFile
from OSmOSE.data.file_base import FileBase
from OSmOSE.data.item_base import ItemBase

if TYPE_CHECKING:
    from pandas import Timestamp


class AudioItem(ItemBase[AudioFile]):
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
    def sample_rate(self) -> float:
        """Sample rate of the associated AudioFile."""
        return None if self.is_empty else self.file.metadata.samplerate

    @property
    def nb_channels(self) -> int:
        """Number of channels in the associated AudioFile."""
        return 0 if self.is_empty else self.file.metadata.channels

    @classmethod
    def from_base_item(cls, item: ItemBase) -> AudioItem:
        file = item.file
        if not file or isinstance(file, AudioFile):
            return cls(file=file, begin=item.begin, end=item.end)
        if isinstance(file, FileBase):
            return cls(
                file=AudioFile.from_base_file(file), begin=item.begin, end=item.end
            )
        raise TypeError
