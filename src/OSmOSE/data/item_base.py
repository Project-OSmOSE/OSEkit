"""ItemBase: Base class for the Item objects (e.g. AudioItem).

Items correspond to a portion of a File object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from pandas import Timestamp

    from OSmOSE.data.file_base import FileBase


class ItemBase:
    """Base class for the Item objects (e.g. AudioItem).

    An Item correspond to a portion of a File object.
    """

    def __init__(
        self,
        file: FileBase,
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
        self.begin = (
            max(begin, self.file.begin) if begin is not None else self.file.begin
        )
        self.end = min(end, self.file.end) if end is not None else self.file.end

    def get_value(self) -> np.ndarray:
        """Get the values from the File between the begin and stop timestamps."""
        return self.file.read(start=self.begin, stop=self.end)
