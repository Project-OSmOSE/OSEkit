"""ItemBase: Base class for the Item objects (e.g. AudioItem).

Items correspond to a portion of a File object.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from OSmOSE.data.file_base import FileBase
from OSmOSE.utils.timestamp_utils import is_overlapping

if TYPE_CHECKING:
    from pandas import Timestamp

TFile = TypeVar("TFile", bound=FileBase)


class ItemBase(Generic[TFile]):
    """Base class for the Item objects (e.g. AudioItem).

    An Item correspond to a portion of a File object.
    """

    def __init__(
        self,
        file: TFile | None = None,
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

    @property
    def total_seconds(self) -> float:
        """Return the total duration of the item in seconds."""
        return (self.end - self.begin).total_seconds()

    def __eq__(self, other: ItemBase[TFile]) -> bool:
        """Override the default implementation."""
        if not isinstance(other, ItemBase):
            return False
        if self.file != other.file:
            return False
        if self.begin != other.begin:
            return False
        return not self.end != other.end

    @staticmethod
    def remove_overlaps(items: list[ItemBase[TFile]]) -> list[ItemBase[TFile]]:
        """Resolve overlaps between Items.

        If two Items overlap within the sequence (that is if one Item begins before the end of another,
        the earliest Item's end is set to the begin of the latest Item.
        If multiple items overlap with one earlier Item, only one is chosen as next.
        The chosen next Item is the one that ends the latest.

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
        >>> items = ItemBase.remove_overlaps(items)
        >>> items[0].end == items[1].begin
        True

        """
        items = sorted(
            [copy.copy(item) for item in items],
            key=lambda item: (item.begin, item.begin - item.end),
        )
        concatenated_items = []
        for item in items:
            concatenated_items.append(item)
            overlapping_items = [
                item2
                for item2 in items
                if item2 is not item
                and is_overlapping((item.begin, item.end), (item2.begin, item2.end))
            ]
            if not overlapping_items:
                continue
            kept_overlapping_item = max(overlapping_items, key=lambda item: item.end)
            if kept_overlapping_item.end > item.end:
                item.end = kept_overlapping_item.begin
            else:
                kept_overlapping_item = None
            for dismissed_item in (
                item2
                for item2 in overlapping_items
                if item2 is not kept_overlapping_item
            ):
                items.remove(dismissed_item)
        return concatenated_items

    @staticmethod
    def fill_gaps(items: list[ItemBase[TFile]]) -> list[ItemBase[TFile]]:
        """Return a list with empty items added in the gaps between items.

        Parameters
        ----------
        items: list[ItemBase]
            List of Items to fill.

        Returns
        -------
        list[ItemBase]:
            List of Items with no gaps.

        Examples
        --------
        >>> items = [ItemBase(begin = Timestamp("00:00:00"), end = Timestamp("00:00:10")), ItemBase(begin = Timestamp("00:00:15"), end = Timestamp("00:00:25"))]
        >>> items = ItemBase.fill_gaps(items)
        >>> [(item.begin.second, item.end.second) for item in items]
        [(0, 10), (10, 15), (15, 25)]

        """
        items = sorted([copy.copy(item) for item in items], key=lambda item: item.begin)
        filled_item_list = []
        for index, item in enumerate(items[:-1]):
            next_item = items[index + 1]
            filled_item_list.append(item)
            if next_item.begin > item.end:
                filled_item_list.append(ItemBase(begin=item.end, end=next_item.begin))
        filled_item_list.append(items[-1])
        return filled_item_list
