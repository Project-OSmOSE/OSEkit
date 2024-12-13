"""DataBase: Base class for the Data objects (e.g. AudioData).

Data corresponds to data scattered through different AudioFiles.
The data is accessed via an AudioItem object per AudioFile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from OSmOSE.data.file_base import FileBase
from OSmOSE.data.item_base import ItemBase
from OSmOSE.utils.timestamp_utils import is_overlapping

if TYPE_CHECKING:
    from pandas import Timestamp


TItem = TypeVar("TItem", bound=ItemBase)
TFile = TypeVar("TFile", bound=FileBase)


class DataBase(Generic[TItem, TFile]):
    """Base class for Data objects.

    A Data object is a collection of Item objects.
    Data can be retrieved from several Files through the Items.
    """

    def __init__(self, items: list[TItem]) -> None:
        """Initialize an DataBase from a list of Items.

        Parameters
        ----------
        items: list[ItemBase]
            List of the Items constituting the Data.

        """
        self.items = items
        self.begin = min(item.begin for item in self.items)
        self.end = max(item.end for item in self.items)

    @property
    def total_seconds(self) -> float:
        """Return the total duration of the data in seconds."""
        return (self.end - self.begin).total_seconds()

    def get_value(self) -> np.ndarray:
        """Get the concatenated values from all Items."""
        return np.concatenate([item.get_value() for item in self.items])

    @classmethod
    def from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> DataBase[TItem, TFile]:
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
        DataBase[TItem, TFile]:
        The DataBase object.

        """
        items = cls.items_from_files(files, begin, end)
        return cls(items)

    @classmethod
    def items_from_files(
        cls,
        files: list[TFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> list[ItemBase]:
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
        list[ItemBase]
            The list of Items that point to the files.

        """
        begin = min(file.begin for file in files) if begin is None else begin
        end = max(file.end for file in files) if end is None else end

        included_files = [
            file
            for file in files
            if is_overlapping((file.begin, file.end), (begin, end))
        ]

        items = [ItemBase(file, begin, end) for file in included_files]
        if not items:
            items.append(ItemBase(begin=begin, end=end))
        if (first_item := sorted(items, key=lambda item: item.begin)[0]).begin > begin:
            items.append(ItemBase(begin=begin, end=first_item.begin))
        if (last_item := sorted(items, key=lambda item: item.end)[-1]).end < end:
            items.append(ItemBase(begin=last_item.end, end=end))
        items = ItemBase.remove_overlaps(items)
        return ItemBase.fill_gaps(items)
