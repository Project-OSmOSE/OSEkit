"""BaseData: Base class for the Data objects.

Data corresponds to data scattered through different Files.
The data is accessed via an Item object per File.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from OSmOSE.data.base_file import BaseFile
from OSmOSE.data.base_item import BaseItem
from OSmOSE.utils.data_utils import EventClass, is_overlapping, remove_overlaps

if TYPE_CHECKING:
    from pandas import Timestamp


TItem = TypeVar("TItem", bound=BaseItem)
TFile = TypeVar("TFile", bound=BaseFile)


class BaseData(Generic[TItem, TFile]):
    """Base class for the Data objects.

    Data corresponds to data scattered through different Files.
    The data is accessed via an Item object per File.
    """

    def __init__(self, items: list[TItem]) -> None:
        """Initialize an BaseData from a list of Items.

        Parameters
        ----------
        items: list[BaseItem]
            List of the Items constituting the Data.

        """
        self.items = items
        self.begin = min(item.begin for item in self.items)
        self.end = max(item.end for item in self.items)

    @property
    def total_seconds(self) -> float:
        """Return the total duration of the data in seconds."""
        return (self.end - self.begin).total_seconds()

    @property
    def is_empty(self) -> bool:
        """Return true if every item of this data object is empty."""
        return all(item.is_empty for item in self.items)

    def get_value(self) -> np.ndarray:
        """Get the concatenated values from all Items."""
        return np.concatenate([item.get_value() for item in self.items])

    def write(self, path: Path) -> None:
        """Abstract method for writing the data."""
        return

    @classmethod
    def from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> BaseData[TItem, TFile]:
        """Return a base DataBase object from a list of Files.

        Parameters
        ----------
        files: list[TFile]
            List of Files containing the data.
        begin: Timestamp | None
            Begin of the data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the data object.
            Defaulted to the end of the last file.

        Returns
        -------
        BaseData[TItem, TFile]:
        The BaseData object.

        """
        items = cls.items_from_files(files, begin, end)
        return cls(items)

    @classmethod
    def items_from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> list[BaseItem]:
        """Return a list of Items from a list of Files and timestamps.

        The Items range from begin to end.
        They point to the files that match their timestamps.

        Parameters
        ----------
        files: list[TFile]
            The Files encapsulated in the Data object.
        begin: pandas.Timestamp | None
            The begin of the Data object.
            defaulted to the begin of the first File.
        end: pandas.Timestamp | None
            The end of the Data object.
            defaulted to the end of the last File.

        Returns
        -------
        list[BaseItem]
            The list of Items that point to the files.

        """
        begin = min(file.begin for file in files) if begin is None else begin
        end = max(file.end for file in files) if end is None else end

        included_files = [
            file
            for file in files
            if is_overlapping(file, EventClass(begin=begin, end=end))
        ]

        items = [BaseItem(file, begin, end) for file in included_files]
        if not items:
            items.append(BaseItem(begin=begin, end=end))
        if (first_item := sorted(items, key=lambda item: item.begin)[0]).begin > begin:
            items.append(BaseItem(begin=begin, end=first_item.begin))
        if (last_item := sorted(items, key=lambda item: item.end)[-1]).end < end:
            items.append(BaseItem(begin=last_item.end, end=end))
        items = remove_overlaps(items)
        return BaseItem.fill_gaps(items)
