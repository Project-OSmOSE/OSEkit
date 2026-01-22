"""``BaseItem``: Base class for the Item objects.

Items correspond to a portion of a File object.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from osekit.core_api.base_file import BaseFile
from osekit.core_api.event import Event

if TYPE_CHECKING:
    from pandas import Timestamp

TFile = TypeVar("TFile", bound=BaseFile)


class BaseItem[TFile: BaseFile](Event, ABC):
    """Base class for the Item objects.

    An Item correspond to a portion of a File object.
    """

    def __init__(
        self,
        file: TFile | None = None,
        begin: Timestamp | None = None,
        end: Timestamp | None = None,
    ) -> None:
        """Initialize an ``BaseItem`` from a File and begin/end timestamps.

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

        begin = max(begin, self.file.begin) if begin is not None else self.file.begin
        end = min(end, self.file.end) if end is not None else self.file.end

        super().__init__(begin=begin, end=end)

    def get_value(self) -> np.ndarray:
        """Get the values from the File between the begin and stop timestamps.

        If the Item is empty, return a single ``0.``.
        """
        return (
            np.zeros(1)
            if self.is_empty
            else self.file.read(start=self.begin, stop=self.end)
        )

    @property
    def is_empty(self) -> bool:
        """Return ``True`` if no File is attached to this Item."""
        return self.file is None

    def __eq__(self, other: BaseItem[TFile]) -> bool:
        """Override the default implementation."""
        if not isinstance(other, BaseItem):
            return False
        if self.file != other.file:
            return False
        if self.begin != other.begin:
            return False
        return not self.end != other.end
