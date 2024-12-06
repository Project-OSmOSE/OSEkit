"""DataBase: Base class for the Data objects (e.g. AudioData).

Data corresponds to a collection of Items.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from OSmOSE.data.item_base import ItemBase
from OSmOSE.utils.timestamp_utils import is_overlapping

if TYPE_CHECKING:
    from pandas import Timestamp

    from OSmOSE.data.file_base import FileBase


class DataBase:
    """Base class for Data objects.

    A Data object is a collection of Item objects.
    Data can be retrieved from several Files through the Items.
    """

    item_cls = ItemBase

    def __init__(self, items: list[ItemBase]) -> None:
        """Initialize an DataBase from a list of Items.

        Parameters
        ----------
        items: list[ItemBase]
            List of the Items constituting the Data.

        """
        self.items = items
        self.begin = min(item.begin for item in self.items)
        self.end = max(item.end for item in self.items)

    def get_value(self) -> np.ndarray:
        """Get the concatenated values from all Items."""
        return np.concatenate([item.get_value() for item in self.items])

    @classmethod
    def from_files(
        cls,
        files: list[FileBase],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> DataBase:
        """Initialize a DataBase from a single File.

        The resulting Data object will contain a single Item.
        This single Item will correspond to the whole File.

        Parameters
        ----------
        files: list[OSmOSE.data.file_base.FileBase]
            The Files encapsulated in the Data object.
        begin: pandas.Timestamp | None
            The begin of the Data object.
            defaulted to the begin of the first File.
        end: pandas.Timestamp | None
            The end of the Data object.
            default to the end of the last File.

        Returns
        -------
        OSmOSE.data.data_base.DataBase
            The Data object.

        """
        begin = min(file.begin for file in files) if begin is None else begin
        end = max(file.end for file in files) if end is None else end

        overlapping_files = [
            file
            for file in files
            if is_overlapping((file.begin, file.end), (begin, end))
        ]

        items = [cls.item_cls(file, begin, end) for file in overlapping_files]
        items = ItemBase.concatenate_items(items)
        items = ItemBase.fill_gaps(items)
        return cls(items=items)
