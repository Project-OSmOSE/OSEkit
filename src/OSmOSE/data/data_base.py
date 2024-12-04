"""DataBase: Base class for the Data objects (e.g. AudioData).

Data corresponds to a collection of Items.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from OSmOSE.data.item_base import ItemBase

if TYPE_CHECKING:
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

    def get_value(self) -> np.ndarray:
        """Get the concatenated values from all Items."""
        return np.concatenate([item.get_value() for item in self.items])

    @classmethod
    def from_file(cls, file: FileBase) -> DataBase:
        """Initialize a DataBase from a single File.

        The resulting Data object will contain a single Item.
        This single Item will correspond to the whole File.

        Parameters
        ----------
        file: OSmOSE.data.file_base.FileBase
            The File encapsulated in the Data object.

        Returns
        -------
        OSmOSE.data.data_base.DataBase
            The Data object.

        """
        item = cls.item_cls(file)
        return cls(items=[item])
