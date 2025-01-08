"""SpectroItem corresponding to a portion of a SpectroFile object."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pandas import DataFrame

from OSmOSE.data.base_file import BaseFile
from OSmOSE.data.base_item import BaseItem
from OSmOSE.data.spectro_file import SpectroFile

if TYPE_CHECKING:
    from pandas import Timedelta, Timestamp


class SpectroItem(BaseItem[SpectroFile]):
    """SpectroItem corresponding to a portion of a SpectroFile object."""

    def __init__(
        self,
        file: SpectroFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> None:
        """Initialize a SpectroItem from a SpectroFile and begin/end timestamps.

        Parameters
        ----------
        file: OSmOSE.data.spectro_file.SpectroFile
            The SpectroFile in which this Item belongs.
        begin: pandas.Timestamp (optional)
            The timestamp at which this item begins.
            It is defaulted to the SpectroFile begin.
        end: pandas.Timestamp (optional)
            The timestamp at which this item ends.
            It is defaulted to the SpectroFile end.

        """
        super().__init__(file, begin, end)
        # self.shape = self.get_value().shape

    @property
    def time_resolution(self) -> Timedelta:
        """Time resolution of the associated SpectroFile."""
        return None if self.is_empty else self.file.time_resolution

    @classmethod
    def from_base_item(cls, item: BaseItem) -> SpectroItem:
        """Return a SpectroItem object from a BaseItem object."""
        file = item.file
        if not file or isinstance(file, SpectroFile):
            return cls(file=file, begin=item.begin, end=item.end)
        if isinstance(file, BaseFile):
            return cls(
                file=SpectroFile.from_base_file(file),
                begin=item.begin,
                end=item.end,
            )
        raise TypeError

    def get_value(self, freq: np.ndarray | None = None) -> DataFrame:
        """Get the values from the File between the begin and stop timestamps.

        If the Item is empty, return a single 0.
        """
        return (
            DataFrame(columns=["time", *freq])
            if self.is_empty
            else self.file.read(start=self.begin, stop=self.end)
        )
