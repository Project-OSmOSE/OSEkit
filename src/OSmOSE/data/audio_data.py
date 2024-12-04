"""AudioData encapsulating to a collection of AudioItem objects."""

from __future__ import annotations

from OSmOSE.data.audio_item import AudioItem
from OSmOSE.data.data_base import DataBase


class AudioData(DataBase):
    """AudioData encapsulating to a collection of AudioItem objects.

    The audio data can be retrieved from several Files through the Items.
    """

    item_cls = AudioItem

    def __init__(self, items: list[AudioItem]) -> None:
        """Initialize an AudioData from a list of AudioItems.

        Parameters
        ----------
        items: list[AudioItem]
            List of the AudioItem constituting the AudioData.

        """
        super().__init__(items)
