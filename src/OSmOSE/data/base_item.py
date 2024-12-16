"""BaseItem: Base class for the Item objects.

Items correspond to a portion of a File object.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from OSmOSE.data.base_file import BaseFile

if TYPE_CHECKING:
    from pandas import Timestamp

TFile = TypeVar("TFile", bound=BaseFile)


class BaseItem(Generic[TFile]):
    """Base class for the Item objects.

    An Item correspond to a portion of a File object.
    """

    def __init__(
        self,
        file: TFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> None:
        """Initialize an BaseItem from a File and begin/end timestamps.

        Parameters
        ----------
        file: TFile
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

    def __eq__(self, other: BaseItem[TFile]) -> bool:
        """Override the default implementation."""
        if not isinstance(other, BaseItem):
            return False
        if self.file != other.file:
            return False
        if self.begin != other.begin:
            return False
        return not self.end != other.end

    @staticmethod
    def fill_gaps(items: list[BaseItem[TFile]]) -> list[BaseItem[TFile]]:
        """Return a list with empty items added in the gaps between items.

        Parameters
        ----------
        items: list[BaseItem]
            List of Items to fill.

        Returns
        -------
        list[BaseItem]:
            List of Items with no gaps.

        Examples
        --------
        >>> items = [BaseItem(begin = Timestamp("00:00:00"), end = Timestamp("00:00:10")), BaseItem(begin = Timestamp("00:00:15"), end = Timestamp("00:00:25"))]
        >>> items = BaseItem.fill_gaps(items)
        >>> [(item.begin.second, item.end.second) for item in items]
        [(0, 10), (10, 15), (15, 25)]

        """
        items = sorted([copy.copy(item) for item in items], key=lambda item: item.begin)
        filled_item_list = []
        for index, item in enumerate(items[:-1]):
            next_item = items[index + 1]
            filled_item_list.append(item)
            if next_item.begin > item.end:
                filled_item_list.append(BaseItem(begin=item.end, end=next_item.begin))
        filled_item_list.append(items[-1])
        return filled_item_list
