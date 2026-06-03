"""The Annotation class represents an annotation made on APLOSE."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from pandas import Timestamp

from osekit.core.event import Event

KNOWN_KEYS = {
    "dataset",
    "project",
    "filename",
    "annotation_id",
    "is_update_of_id",
    "start_time",
    "end_time",
    "start_frequency",
    "end_frequency",
    "min_frequency",
    "max_frequency",
    "annotation",
    "annotator",
    "annotator_expertise",
    "start_datetime",
    "end_datetime",
    "is_box",
    "type",
    "label",
    "level",
    "comments",
    "signal_quantity",
    "signal_is_intensity_too_low",
    "signal_does_overlap_other_signals",
    "signal_start_frequency",
    "signal_end_frequency",
    "signal_relative_min_frequency_count",
    "signal_relative_max_frequency_count",
    "signal_steps_count",
    "signal_has_harmonics",
    "signal_trend",
    "signal_sidebands",
    "signal_subharmonics",
    "signal_frequency_jumps",
    "signal_deterministic_chaos",
    "created_at_phase",
}


@dataclass
class FrequencyBounds:
    """Class representing  the frequency bounds of an annotation.

    Parameters
    ----------
    min: int
        Lower frequency bound.
    max: int
        Upper frequency bound.

    """

    min: int
    max: int

    def __post_init__(self) -> None:
        """Check the validity of the frequency bounds."""
        error_msgs = []
        if self.min < 0:
            error_msgs.append(
                f"Min frequency must be greater than or equal to 0, got {self.min}.",
            )
        if self.max < 0:
            error_msgs.append(
                f"Max frequency must be greater than or equal to 0, got {self.max}.",
            )
        if self.min > self.max:
            error_msgs.append(
                f"Max frequency must be greater than min frequency, "
                f"got ({self.min},{self.max}).",
            )
        if error_msgs:
            msg = "\n".join(error_msgs)
            raise ValueError(msg)

    @property
    def bandwidth(self) -> int:
        """Bandwidth of the annotation."""
        return self.max - self.min


@dataclass
class AnnotatorInfo:
    """Class representing an annotator info."""

    annotator: str
    annotator_expertise: Literal["NOVICE", "AVERAGE", "EXPERT"] | None = None

    def __hash__(self) -> int:
        """Return a hash for the annotator."""
        return hash((self.annotator, self.annotator_expertise))


@dataclass
class SignalParameters:
    """Class representing parameters of an annoted signal."""

    is_itensity_too_low: bool | None = None
    does_overlap_other_signals: bool | None = None
    min_frequency: int | None = None
    max_frequency: int | None = None
    nb_relative_mins: int | None = None
    nb_relative_maxes: int | None = None
    nb_steps: int | None = None
    trend: Literal["FLAT", "ASCENDING", "DESCENDING", "MODULATED"] | None = None
    frequency_jumps: bool | int | None = None
    has_harmonics: bool | None = None
    has_sidebands: bool | None = None
    has_subharmonics: bool | None = None
    has_deterministic_chaos: bool | None = None


@dataclass
class ConfidenceIndicator:
    """Class that represents an annotation confidence indicator.

    Parameters
    ----------
    label: str
        Name of the level of confidence.
    level: int
        Level of confidence of the annotation.
    maximum_level: int
        Maximum level of confidence authorized in the project.

    """

    label: str
    level: int
    maximum_level: int

    def __post_init__(self) -> None:
        """Check the validity of the level and maximum level values."""
        if self.level > self.maximum_level:
            msg = (
                f"Confidence level {self.level} is higher than "
                f"maximum level {self.maximum_level} authorized in the project."
            )
            raise ValueError(msg)

    @classmethod
    def from_relative_level_string(cls, label: str, relative_level_string: str) -> Self:
        """Return a ``ConfidenceIndicator`` from a string representing its level.

        Parameters
        ----------
        label: str
            Name of the level of confidence.
        relative_level_string: str
            Level of confidence relative to the maximum level available.
            Should be formatted as ``n/m``, where ``n`` is the level of confidence
            of the annotation and ``m`` is the maximum level available in the project.

        Returns
        -------
        ConfidenceIndicator
            The confidence indicator parsed from the input string.

        """
        level, maximum_level = map(int, relative_level_string.split("/"))

        return cls(label=label, level=level, maximum_level=maximum_level)


@dataclass
class AnnotationMetaData:
    """Class that represents the metadata of an annotation.

    Parameters
    ----------
    project: str
        Name of the project in which the annotation was made.
    filename: str
        Name of the file this annotation was made on.
    annotation_id: int
        ID of the annotation.
    base_id: int
        ID of the base annotation.
        May differ from ``annotation_id`` if the annotation is an update/correction.
    comments: str | None
        Comments left by the annotator.
    phase: Literal["ANNOTATION", "VERIFICATION"]
        Phase during which the annotation was created.

    """

    project: str
    filename: str
    annotation_id: int
    base_id: int | None
    comments: str | None
    phase: Literal["ANNOTATION", "VERIFICATION"]


@dataclass
class Verification:
    """Class that represents a verification of an annotation."""

    verificator: str
    is_validated: bool


class Annotation(Event):
    """Class that represents an annotation made on APLOSE."""

    def __init__(  # noqa: PLR0913
        self,
        metadata: AnnotationMetaData,
        begin: Timestamp,
        end: Timestamp,
        frequency_bounds: FrequencyBounds,
        label: str,
        annotator_info: AnnotatorInfo,
        annotation_type: Literal["WEAK", "POINT", "BOX"],
        confidence_indicator: ConfidenceIndicator,
        signal_quantity: Literal["SINGLE", "MULTIPLE"],
        signal_parameters: SignalParameters | None,
        verifications: list[Verification],
    ) -> None:
        """Initialize an Annotation object.

        Parameters
        ----------
        metadata: AnnotationMetaData
            Metadata on the annotation.
        begin: Timestamp
            Begin timestamp of the annotation.
        end: Timestamp
            End timestamp of the annotation.
        frequency_bounds: FrequencyBounds
            Frequency bounds of the annotation.
        label: str
            Label of the annotation.
        annotator_info: AnnotatorInfo
            Information on the annotator or detector.
        annotation_type: Literal["WEAK", "POINT", "BOX"]
            Type of the annotation.
            ``WEAK``: Annotation made on the whole spectrogram.
            ``POINT``: Annotation made on one pixel of the spectrogram.
            ``BOX``: Annotation made on one box within the spectrogram.
        confidence_indicator: ConfidenceIndicator
            Indicator of the confidence of the annotator.
        signal_quantity: Literal["SINGLE","MULTIPLE"]
            Whether there is only one signal in the annotation or more.
        signal_parameters: SignalParameters | None
            Parameters of the annotated signal.
            ```None`` if ``signal_quantity`` is ``MULTIPLE``.
        verifications: list[Verification]
            Verifications made on this annotation.

        """
        self.metadata = metadata
        self.label = label
        self.annotator_info = annotator_info
        self.frequency_bounds = frequency_bounds
        self.type = annotation_type
        self.confidence_indicator = confidence_indicator
        self.signal_quantity = signal_quantity
        self.signal_parameters = signal_parameters
        self.verifications = verifications

        super().__init__(begin=begin, end=end)

    def __repr__(self) -> str:
        """Override the string representation of the annotation."""
        return str(self.metadata.annotation_id)

    @classmethod
    def from_dict(cls, row: dict) -> Self:
        """Deserialize an Annotation object."""
        metadata = AnnotationMetaData(
            project=row["project"] if "project" in row else row["dataset"],
            filename=row["filename"],
            annotation_id=row["annotation_id"],
            base_id=row["is_update_of_id"],
            comments=row["comments"],
            phase=row["created_at_phase"],
        )
        annotator_info = AnnotatorInfo(
            annotator=row["annotator"],
            annotator_expertise=row["annotator_expertise"],
        )

        min_frequency, max_frequency = row["min_frequency"], row["max_frequency"]
        frequency_bounds = (
            FrequencyBounds(min=min_frequency, max=max_frequency)
            if not any(m is None for m in (min_frequency, max_frequency))
            else None
        )

        confidence_indicator = ConfidenceIndicator.from_relative_level_string(
            label=row["confidence_indicator_label"],
            relative_level_string=row["confidence_indicator_level"],
        )

        signal_quantity = row["signal_quantity"]
        signal_parameters = (
            SignalParameters(
                does_overlap_other_signals=row["signal_is_intensity_too_low"],
                frequency_jumps=row["signal_frequency_jumps"],
                has_deterministic_chaos=row["signal_deterministic_chaos"],
                has_harmonics=row["signal_has_harmonics"],
                has_sidebands=row["signal_sidebands"],
                has_subharmonics=row["signal_subharmonics"],
                is_itensity_too_low=row["signal_is_intensity_too_low"],
                max_frequency=row["signal_end_frequency"],
                min_frequency=row["signal_start_frequency"],
                nb_relative_maxes=row["signal_relative_max_frequency_count"],
                nb_relative_mins=row["signal_relative_min_frequency_count"],
                nb_steps=row["signal_steps_count"],
                trend=row["signal_trend"],
            )
            if signal_quantity == "SINGLE"
            else None
        )

        verifications = [
            Verification(
                verificator=key,
                is_validated=value,
            )
            for key, value in row.items()
            if key not in KNOWN_KEYS
        ]

        return cls(
            metadata=metadata,
            label=row["annotation"],
            annotator_info=annotator_info,
            begin=Timestamp(row["start_datetime"]),
            end=Timestamp(row["end_datetime"]),
            frequency_bounds=frequency_bounds,
            annotation_type=row["type"],
            confidence_indicator=confidence_indicator,
            signal_quantity=row["signal_quantity"],
            signal_parameters=signal_parameters,
            verifications=verifications,
        )

    @classmethod
    def from_csv(cls, csv: Path) -> list[Self]:
        """Deserialize a list of Annotation from an annotations csv file."""
        records = pd.read_csv(filepath_or_buffer=csv).to_dict(
            orient="records",
        )
        records = [
            {
                key: None if type(value) is float and math.isnan(value) else value
                for key, value in record.items()
            }
            for record in records
        ]
        return [cls.from_dict(record) for record in records]
