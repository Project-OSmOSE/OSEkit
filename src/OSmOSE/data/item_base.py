"""ItemBase: Base class for the Item objects (e.g. AudioItem).

Items correspond to a portion of a File object.
"""

from __future__ import annotations

import numpy as np
from pandas import Timestamp

from OSmOSE.data.file_base import FileBase
from OSmOSE.utils.timestamp_utils import is_overlapping


class ItemBase:
    """Base class for the Item objects (e.g. AudioItem).

    An Item correspond to a portion of a File object.
    """

    def __init__(
        self,
        file: FileBase | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> None:
        """Initialize an ItemBase from a File and begin/end timestamps.

        Parameters
        ----------
        file: OSmOSE.data.file_base.FileBase
            The File in which this Item belongs.
        begin: pandas.Timestamp (optional)
            The timestamp at which this item begins.
            It is defaulted or maxed to the File begin.
        end: pandas.Timestamp (optional)
            The timestamp at which this item ends.
            It is defaulted or mined to the File end.

        """
        self.file = file

        if file is None:
            self.begin = begin
            self.end = end
            return

        self.begin = (
            max(begin, self.file.begin) if begin is not None else self.file.begin
        )
        self.end = min(end, self.file.end) if end is not None else self.file.end

    def get_value(self) -> np.ndarray:
        """Get the values from the File between the begin and stop timestamps.

        If the Item is empty, return a single 0.
        """
        return (
            np.zeros(1)
            if self.is_empty
            else self.file.read(start=self.begin, stop=self.end)
        )

    @property
    def is_empty(self) -> bool:
        """Return True if no File is attached to this Item."""
        return self.file is None

    @staticmethod
    def concatenate_items(items: list[ItemBase]) -> list[ItemBase]:
        """Resolve overlaps between Items.

        If two Items overlap within the sequence (that is if one Item begins before the end of another,
        the earliest Item's end is set to the begin of the latest Item.
        If multiple items overlap with one earlier Item, only one is chosen as next.
        The chosen next Item is the one that ends the latest.
        The Items are concatenated in-place.

        Parameters
        ----------
        items: list[ItemBase]
            List of Items to concatenate.

        Returns
        -------
        list[ItemBase]:
            The list of Items with no overlapping Items.

        Examples
        --------
        >>> items = [ItemBase(begin = Timestamp("00:00:00"), end = Timestamp("00:00:15")), ItemBase(begin = Timestamp("00:00:10"), end = Timestamp("00:00:20"))]
        >>> items[0].end == items[1].begin
        False
        >>> ItemBase.concatenate_items(items) # doctest: +ELLIPSIS
        [<src.OSmOSE.data.item_base.ItemBase object ...>, <src.OSmOSE.data.item_base.ItemBase object ...>]
        >>> items[0].end == items[1].begin
        True

        """
        items = sorted(items, key=lambda item: (item.begin, item.end))
        concatenated_items: list[ItemBase] = []
        for item in items:
            overlapping_items = [
                item2
                for item2 in items
                if is_overlapping((item.begin, item.end), (item2.begin, item2.end))
            ]
            if len(overlapping_items) == 1:
                concatenated_items.append(item)
                continue
            kept_overlapping_item = max(overlapping_items, key=lambda item: item.end)
            if kept_overlapping_item is not item:
                item.end = kept_overlapping_item.begin
            concatenated_items.append(item)
            for dismissed_item in (
                item2
                for item2 in overlapping_items
                if item2 not in (item, kept_overlapping_item)
            ):
                items.remove(dismissed_item)
        return concatenated_items
