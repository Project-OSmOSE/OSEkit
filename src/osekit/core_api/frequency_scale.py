"""Custom frequency scales for plotting spectrograms.

The custom scale is formed from a list of ``ScaleParts``, which assign a
frequency range to a range on the scale.
Provided ``ScaleParts`` should cover the whole scale (from 0% to 100%).

Such Scale can then be passed to the ``SpectroData.plot()`` method for the
spectrogram to be plotted on a custom frequency scale.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from osekit.utils.core_utils import get_closest_value_index


@dataclass(frozen=True)
class ScalePart:
    """Represent a part of the frequency scale of a spectrogram.

    p_min: float
        Relative position of the bottom of the scale part on the full scale.
        Must be in the interval ``[0.0, 1.0]``, where ``0.0`` is the bottom of the scale
        and ``1.0`` is the top.
    p_max: float
        Relative position of the top of the scale part on the full scale.
        Must be in the interval ``[0.0, 1.0]``, where ``0.0`` is the bottom of the scale
        and ``1.0`` is the top.
    f_min: float
        Frequency corresponding to the bottom of the scale part.
    f_max: float
        Frequency corresponding to the top of the scale part.
    scale_type: Literal["lin", "log"]
        Type of the scale, either linear or logarithmic.

    """

    p_min: float
    p_max: float
    f_min: float
    f_max: float
    scale_type: Literal["lin", "log"] = "lin"

    def __post_init__(self) -> None:
        """Check if ``ScalePart`` values are correct."""
        err = []
        if not 0.0 <= self.p_min <= 1.0:
            err.append(f"p_min must be between 0 and 1, got {self.p_min}")
        if not 0.0 <= self.p_max <= 1.0:
            err.append(f"p_max must be between 0 and 1, got {self.p_max}")
        if self.p_min >= self.p_max:
            err.append(
                f"p_min must be strictly inferior than p_max, got ({self.p_min},{self.p_max})",
            )
        if self.f_min < 0:
            err.append(
                f"f_min must be positive, got {self.f_min}",
            )
        if self.f_max < 0:
            err.append(
                f"f_max must be positive, got {self.f_max}",
            )
        if self.f_min >= self.f_max:
            err.append(
                f"f_min must be strictly inferior than f_max, got ({self.f_min},{self.f_max})",
            )
        if err:
            msg = "\n".join(err)
            raise ValueError(msg)

    def get_frequencies(self, nb_points: int) -> list[int]:
        """Return the frequency points of the present scale part."""
        space = self.scale_lambda(self.f_min, self.f_max, nb_points)
        return list(map(round, space))

    def get_indexes(self, scale_length: int) -> tuple[int, int]:
        """Return the indexes of the present scale part in the full scale."""
        return int(self.p_min * scale_length), int(self.p_max * scale_length)

    def get_values(self, scale_length: int) -> list[int]:
        """Return the values of the present scale part."""
        start, stop = self.get_indexes(scale_length)
        return list(self.scale_lambda(self.f_min, self.f_max, stop - start))

    def to_dict_value(self) -> tuple[float, float, float, float, str]:
        """Serialize a ScalePart to a dictionary entry."""
        return self.p_min, self.p_max, self.f_min, self.f_max, self.scale_type

    def __eq__(self, other: any) -> bool:
        """Overwrite eq dunder."""
        if type(other) is not ScalePart:
            return False
        return (
            self.p_min == other.p_min
            and self.p_max == other.p_max
            and self.f_min == other.f_min
            and self.f_max == other.f_max
            and self.scale_type == other.scale_type
        )

    @property
    def scale_lambda(self) -> callable:
        """Lambda function used to generate either a linear or logarithmic scale."""
        return lambda start, stop, steps: (
            np.linspace(start, stop, steps)
            if self.scale_type == "lin"
            else np.geomspace(start, stop, steps)
        )


class Scale:
    """Class that represent a custom frequency scale for plotting spectrograms.

    The custom scale is formed from a list of ``ScaleParts``, which assign a
    frequency range to a range on the scale.
    Provided ``ScaleParts`` should cover the whole scale (from ``0%`` to ``100%``).

    Such ``Scale`` can then be passed to the ``SpectroData.plot()`` method for the
    spectrogram to be plotted on a custom frequency scale.

    """

    def __init__(self, parts: list[ScalePart]) -> None:
        """Initialize a ``Scale`` object."""
        self.parts = sorted(parts, key=lambda p: (p.p_min, p.p_max))

    def map(self, original_scale_length: int) -> list[float]:
        """Map a given scale to the custom scale defined by its ``ScaleParts``.

        Parameters
        ----------
        original_scale_length: int
            Length of the original frequency scale.

        Returns
        -------
        list[float]
            Mapped frequency scale.
            Each ``ScalePart`` from the ``Scale.parts`` attribute are concatenated
            to form the returned scale.

        """
        return [
            v for scale in self.parts for v in scale.get_values(original_scale_length)
        ]

    def get_mapped_indexes(self, original_scale: list[float]) -> list[int]:
        """Return the indexes of the present scale in the original scale.

        The indexes are those of the closest value from the mapped values
        in the original scale.

        Parameters
        ----------
        original_scale: list[float]
            Original scale from which the mapped scale is computed.

        Returns
        -------
        list[int]
            Indexes of the closest value from the mapped values in the
            original scale.

        """
        mapped_scale = self.map(len(original_scale))
        return [
            get_closest_value_index(target=mapped, values=original_scale)
            for mapped in mapped_scale
        ]

    def get_mapped_values(self, original_scale: list[float]) -> list[float]:
        """Return the closest values of the mapped scale from the original scale.

        Parameters
        ----------
        original_scale: list[float]
            Original scale from which the mapped scale is computed.

        Returns
        -------
        list[float]
            Values from the original scale that are the closest to the mapped scale.

        """
        return [original_scale[i] for i in self.get_mapped_indexes(original_scale)]

    def rescale(
        self,
        sx_matrix: np.ndarray,
        original_scale: np.ndarray | list,
    ) -> np.ndarray:
        """Rescale the given spectrum matrix according to the present scale.

        Parameters
        ----------
        sx_matrix: np.ndarray
            Spectrum matrix.
        original_scale: np.ndarray
            Original frequency axis of the spectrum matrix.

        Returns
        -------
        np.ndarray
            Spectrum matrix mapped on the present scale.

        """
        if type(original_scale) is np.ndarray:
            original_scale = original_scale.tolist()

        new_scale_indexes = self.get_mapped_indexes(original_scale=original_scale)

        return sx_matrix[new_scale_indexes]

    def to_dict_value(self) -> list[tuple[float, float, float, float, str]]:
        """Serialize a ``Scale`` to a dictionary entry."""
        return [part.to_dict_value() for part in self.parts]

    @classmethod
    def from_dict_value(cls, dict_value: list[list]) -> Scale:
        """Deserialize a ``Scale`` from a dictionary entry."""
        return cls([ScalePart(*scale) for scale in dict_value])

    def __eq__(self, other: any) -> bool:
        """Overwrite eq dunder."""
        if type(other) is not Scale:
            return False
        return self.parts == other.parts
