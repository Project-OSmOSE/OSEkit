"""SpectroData represent spectrogram data retrieved from SpectroFiles.

The SpectroData has a collection of SpectroItem.
The data is accessed via a SpectroItem object per SpectroFile.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from OSmOSE.config import TIMESTAMP_FORMAT_EXPORTED_FILES
from OSmOSE.data.base_data import BaseData
from OSmOSE.data.spectro_file import SpectroFile
from OSmOSE.data.spectro_item import SpectroItem

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import Timedelta, Timestamp


class SpectroData(BaseData[SpectroItem, SpectroFile]):
    """SpectroData represent Spectro data scattered through different SpectroFiles.

    The SpectroData has a collection of SpectroItem.
    The data is accessed via a SpectroItem object per SpectroFile.
    """

    def __init__(
        self,
        items: list[SpectroItem] | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
        time_resolution: Timedelta | None = None,
    ) -> None:
        """Initialize a SpectroData from a list of SpectroItems.

        Parameters
        ----------
        items: list[SpectroItem]
            List of the SpectroItem constituting the SpectroData.
        time_resolution: Timedelta
            The time resolution of the Spectro data.
        begin: Timestamp | None
            Only effective if items is None.
            Set the begin of the empty data.
        end: Timestamp | None
            Only effective if items is None.
            Set the end of the empty data.

        """
        super().__init__(items=items, begin=begin, end=end)
        self._set_time_resolution(time_resolution=time_resolution)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the Spectro data."""
        return max(item.shape[0] for item in self.items), sum(
            item.shape[1] for item in self.items
        )

    def __str__(self) -> str:
        """Overwrite __str__."""
        return self.begin.strftime(TIMESTAMP_FORMAT_EXPORTED_FILES)

    def _set_time_resolution(self, time_resolution: Timedelta) -> None:
        """Set the SpectroFile time resolution."""
        if len(tr := {item.time_resolution for item in self.items}) > 1:
            raise ValueError("Items don't have the same time resolution")
        self.time_resolution = tr.pop() if len(tr) == 1 else time_resolution

    def get_value(self) -> np.ndarray:
        """Return the value of the Spectro data.

        The data from the Spectro file will be resampled if necessary.
        """
        data = np.zeros(shape=self.shape)
        idx = 0
        for item in self.items:
            item_data = self._get_item_value(item)
            time_bins = item_data.shape[1]
            data[:, idx : idx + time_bins] = item_data
            idx += time_bins
        return data

    def write(self, folder: Path) -> None:
        """Write the Spectro data to file.

        Parameters
        ----------
        folder: pathlib.Path
            Folder in which to write the Spectro file.

        """
        super().write(path=folder)
        # TODO: implement npz write

    def _get_item_value(self, item: SpectroItem) -> np.ndarray:
        """Return the resampled (if needed) data from the Spectro item."""
        item_data = item.get_value()
        if item.is_empty:
            return item_data.repeat(round(item.duration / self.time_resolution))
        if item.time_resolution != self.time_resolution:
            raise ValueError("Time resolutions don't match.")
        return item_data

    @classmethod
    def from_files(
        cls,
        files: list[SpectroFile],
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> SpectroData:
        """Return a SpectroData object from a list of SpectroFiles.

        Parameters
        ----------
        files: list[SpectroFile]
            List of SpectroFiles containing the data.
        begin: Timestamp | None
            Begin of the data object.
            Defaulted to the begin of the first file.
        end: Timestamp | None
            End of the data object.
            Defaulted to the end of the last file.

        Returns
        -------
        SpectroData:
            The SpectroData object.

        """
        return cls.from_base_data(BaseData.from_files(files, begin, end))

    @classmethod
    def from_base_data(
        cls,
        data: BaseData,
    ) -> SpectroData:
        """Return an SpectroData object from a BaseData object.

        Parameters
        ----------
        data: BaseData
            BaseData object to convert to SpectroData.

        Returns
        -------
        SpectroData:
            The SpectroData object.

        """
        return cls([SpectroItem.from_base_item(item) for item in data.items])
