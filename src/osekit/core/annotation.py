"""The Annotation class represents an annotation made on APLOSE."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import pandas as pd
from pandas import Timestamp

from osekit.core.event import Event

KNOWN_KEYS = {
    "dataset",
    "analysis",
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
    "confidence_indicator_label",
    "confidence_indicator_level",
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

    @property
    def bandwidth(self) -> int:
        """Bandwidth of the annotation."""
        return self.max - self.min


@dataclass
class AnnotatorInfo:
    """Class representing an annotator info."""

    annotator: str
    annotator_expertise: Literal["NOVICE", "AVERAGE", "EXPERT"]


@dataclass
class SignalParameters:
    """Class representing parameters of an annoted signal."""

    is_itensity_too_low: bool
    does_overlap_other_signals: bool
    min_frequency: int
    max_frequency: int
    nb_relative_mins: int
    nb_relative_maxes: int
    nb_steps: int
    trend: Literal["FLAT", "ASCENDING", "DESCENDING", "MODULATED"]
    frequency_jumps: bool | int
    has_harmonics: bool
    has_sidebands: bool
    has_subharmonics: bool
    has_deterministic_chaos: bool


@dataclass
class ConfidenceIndicator:
    """Class that represents an annotation confidence indicator.

    Parameters
    ----------
    confidence_indicator_label: str
        Name of the level of confidence.
    confidence_indicator_level: str
        Level of confidence relative to the maximum level available.
        Should be formatted as ``n/m``, where ``n`` is the level of confidence
        of the annotation and ``m`` is the maximum level available in the project.

    """

    confidence_indicator_label: str
    confidence_indicator_level: str


@dataclass
class AnnotationMetaData:
    """Class that represents the metadata of an annotation.

    Parameters
    ----------
    project: str
        Name of the project in which the annotation was made.
    output: str
        Name of the output ``SpectroDataset`` this annotation was made on.
    filename: str
        Name of the file this annotation was made on.
    annotation_id: int
        ID of the annotation.
    base_id: int
        ID of the base annotation.
        May differ from ``id`` if the annotation is an update/correction.

    """

    project: str
    output: str
    filename: str
    annotation_id: int
    base_id: int


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
        comments: str,
        phase: Literal["ANNOTATION", "VERIFICATION"],
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
        comments: str
            Comments left by the annotator.
        phase: Literal["ANNOTATION", "VERIFICATION"]
            Phase during which the annotation was created.
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
        self.comments = comments
        self.phase = phase
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
            output=row["output"] if "output" in row else row["analysis"],
            filename=row["filename"],
            annotation_id=row["annotation_id"],
            base_id=row["is_update_of_id"],
        )
        annotator_info = AnnotatorInfo(
            annotator=row["annotator"],
            annotator_expertise=row["annotator_expertise"],
        )
        frequency_bounds = FrequencyBounds(
            min=row["min_frequency"],
            max=row["max_frequency"],
        )
        confidence_indicator = ConfidenceIndicator(
            confidence_indicator_label=row["confidence_indicator_label"],
            confidence_indicator_level=row["confidence_indicator_level"],
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
            comments=row["comments"],
            phase=row["created_at_phase"],
            signal_quantity=row["signal_quantity"],
            signal_parameters=signal_parameters,
            verifications=verifications,
        )

    @classmethod
    def from_csv(cls, csv: Path) -> list[Self]:
        """Deserialize a list of Annotation from an annotations csv file."""
        return [
            cls.from_dict(record)
            for record in pd.read_csv(filepath_or_buffer=csv).to_dict(orient="records")
        ]
