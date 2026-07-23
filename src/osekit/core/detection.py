"""The Detection class represents a detection made on APLOSE."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd
from matplotlib.patches import Rectangle
from pandas import Timestamp

from osekit.core.event import Event
from osekit.utils.core import is_empty_dataclass

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
    """Class representing  the frequency bounds of a detection.

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
        """Bandwidth of the detection."""
        return self.max - self.min


@dataclass
class DetectorInfo:
    """Class representing a detector info."""

    name: str
    expertise: Literal["NOVICE", "AVERAGE", "EXPERT"] | None = None

    def __hash__(self) -> int:
        """Return a hash for the detector."""
        return hash((self.name, self.expertise))

    def __eq__(self, other: Self) -> bool:
        """Return whether two detectors are equal."""
        return self.name == other.name and self.expertise == other.expertise


@dataclass
class SignalParameters:
    """Class representing parameters of detection signal."""

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
    """Class that represents a detection confidence indicator.

    Parameters
    ----------
    label: str
        Name of the level of confidence.
    level: int
        Level of confidence of the detection.
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
            of the detection and ``m`` is the maximum level available in the project.

        Returns
        -------
        ConfidenceIndicator
            The confidence indicator parsed from the input string.

        """
        level, maximum_level = map(int, relative_level_string.split("/"))

        return cls(label=label, level=level, maximum_level=maximum_level)


@dataclass
class DetectionMetaData:
    """Class that represents the metadata of a detection.

    Parameters
    ----------
    project: str | None
        Name of the project in which the detection was made.
    filename: str | None
        Name of the file this detection was made on.
    detection_id: int | None
        ID of the detection.
    base_id: int | None
        ID of the base detection.
        May differ from ``detection_id`` if the detection is an update/correction.
    comments: str | None
        Comments left by the annotator.
    phase: Literal["ANNOTATION", "VERIFICATION"] | None
        Phase during which the detection was created.

    """

    project: str | None
    filename: str | None
    detection_id: int | None
    base_id: int | None
    comments: str | None
    phase: Literal["ANNOTATION", "VERIFICATION"] | None


@dataclass
class Verification:
    """Class that represents a verification of a detection."""

    verificator: str
    is_validated: bool

    def __hash__(self) -> int:
        """Return a hash of the verification."""
        return hash((self.verificator, self.is_validated))

    def __eq__(self, other: Self) -> bool:
        """Return whether the two verifications are equal."""
        return (
            self.verificator == other.verificator
            and self.is_validated == other.is_validated
        )


class Detection(Event):
    """Class that represents a detection made on APLOSE."""

    def __init__(  # noqa: PLR0913
        self,
        begin: Timestamp,
        end: Timestamp,
        frequency_bounds: FrequencyBounds,
        metadata: DetectionMetaData | None = None,
        label: str | None = None,
        detector_info: DetectorInfo | None = None,
        detection_type: Literal["WEAK", "POINT", "BOX"] | None = None,
        confidence_indicator: ConfidenceIndicator | None = None,
        signal_quantity: Literal["SINGLE", "MULTIPLE"] | None = None,
        signal_parameters: SignalParameters | None = None,
        verifications: set[Verification] | None = None,
    ) -> None:
        """Initialize a Detection object.

        Parameters
        ----------
        begin: Timestamp
            Begin timestamp of the detection.
        end: Timestamp
            End timestamp of the detection.
        frequency_bounds: FrequencyBounds
            Frequency bounds of the detection.
        metadata: DetectionMetaData | None
            Metadata on the detection.
        label: str | None
            Label of the detection.
        detector_info: DetectorInfo | None
            Information on the annotator or detector.
        detection_type: Literal["WEAK", "POINT", "BOX"] | None
            Type of the detection.
            ``WEAK``: Detection made on the whole spectrogram.
            ``POINT``: Detection made on one pixel of the spectrogram.
            ``BOX``: Detection made on one box within the spectrogram.
        confidence_indicator: ConfidenceIndicator | None
            Indicator of the confidence of the annotator.
        signal_quantity: Literal["SINGLE","MULTIPLE"] | None
            Whether there is only one signal in the detection or more.
        signal_parameters: SignalParameters | None
            Parameters of the annotated signal.
            ```None`` if ``signal_quantity`` is ``MULTIPLE``.
        verifications: set[Verification] | None
            Verifications made on this detection.

        """
        self.metadata = metadata
        self.label = label
        self.detector_info = detector_info
        self.frequency_bounds = frequency_bounds
        self.type = detection_type
        self.confidence_indicator = confidence_indicator
        self.signal_quantity = signal_quantity
        self.signal_parameters = signal_parameters
        self.verifications = verifications or {}

        super().__init__(begin=begin, end=end)

    def __repr__(self) -> str:
        """Override the string representation of the detection."""
        return (
            str(self.metadata.detection_id)
            if self.metadata and self.metadata.detection_id
            else f"{self.begin.strftime('%Y-%m-%dT%H:%M:%S%Z')} - "
            f"{self.end.strftime('%Y-%m-%dT%H:%M:%S%Z')} "
            f"[{self.frequency_bounds.min} Hz - {self.frequency_bounds.max} Hz]"
        )

    @classmethod
    def from_dict(cls, row: dict) -> Self:
        """Deserialize a Detection object."""
        metadata = DetectionMetaData(
            project=row.get("project", row.get("dataset")),
            filename=str(row.get("filename")) if row.get("filename") else None,
            detection_id=row.get("annotation_id"),
            base_id=row.get("is_update_of_id"),
            comments=row.get("comments"),
            phase=row.get("created_at_phase"),
        )
        metadata = None if is_empty_dataclass(instance=metadata) else metadata

        detector_info = (
            DetectorInfo(
                name=str(row.get("annotator")),
                expertise=row.get("annotator_expertise"),
            )
            if row.get("annotator")
            else None
        )

        frequency_bounds = FrequencyBounds(
            min=row["min_frequency"],
            max=row["max_frequency"],
        )

        confidence_indicator = (
            ConfidenceIndicator.from_relative_level_string(
                label=str(row.get("confidence_indicator_label")),
                relative_level_string=str(row.get("confidence_indicator_level")),
            )
            if row.get("confidence_indicator_label")
            else None
        )

        signal_quantity = row.get("signal_quantity")
        signal_parameters = (
            SignalParameters(
                does_overlap_other_signals=row.get("signal_is_intensity_too_low"),
                frequency_jumps=row.get("signal_frequency_jumps"),
                has_deterministic_chaos=row.get("signal_deterministic_chaos"),
                has_harmonics=row.get("signal_has_harmonics"),
                has_sidebands=row.get("signal_sidebands"),
                has_subharmonics=row.get("signal_subharmonics"),
                is_itensity_too_low=row.get("signal_is_intensity_too_low"),
                max_frequency=row.get("signal_end_frequency"),
                min_frequency=row.get("signal_start_frequency"),
                nb_relative_maxes=row.get("signal_relative_max_frequency_count"),
                nb_relative_mins=row.get("signal_relative_min_frequency_count"),
                nb_steps=row.get("signal_steps_count"),
                trend=row.get("signal_trend"),
            )
            if signal_quantity == "SINGLE"
            else None
        )

        verifications = {
            Verification(
                verificator=key,
                is_validated=value,
            )
            for key, value in row.items()
            if key not in KNOWN_KEYS
        }
        verifications = {v for v in verifications if v.is_validated is not None}

        return cls(
            metadata=metadata,
            label=row.get("annotation"),
            detector_info=detector_info,
            begin=Timestamp(row["start_datetime"]),
            end=Timestamp(row["end_datetime"]),
            frequency_bounds=frequency_bounds,
            detection_type=row.get("type"),
            confidence_indicator=confidence_indicator,
            signal_quantity=row.get("signal_quantity"),
            signal_parameters=signal_parameters,
            verifications=verifications,
        )

    def to_rectangle(self, *, fill: bool = False, **kwargs: Any) -> Rectangle:
        """Return a matplotlib Rectangle representing the detection.

        Parameters
        ----------
        fill: bool
            Set whether to fill the patch.
            Defaulted to False.
        kwargs:
            Additional keyword arguments

        Returns
        -------
        matplotlib.patches.Rectangle
            Rectangle representing the detection.
            The coordinates of the rectangle are in time x frequency.



        """
        return Rectangle(
            xy=(  # type: ignore[arg-type]
                self.begin,
                self.frequency_bounds.min,
            ),
            width=self.duration,  # type: ignore[arg-type]
            height=self.frequency_bounds.bandwidth,
            fill=fill,
            **kwargs,
        )

    @classmethod
    def _from_csv(cls, csv: Path) -> list[Self]:
        records = (
            pd.read_csv(filepath_or_buffer=csv)
            .convert_dtypes()
            .to_dict(
                orient="records",
            )
        )
        records = [
            {
                key: None if type(value) is float and math.isnan(value) else value
                for key, value in record.items()
            }
            for record in records
        ]
        return [cls.from_dict(record) for record in records]

    @classmethod
    def from_csv(cls, csv: Path | list[Path]) -> list[Self]:
        """Deserialize a list of Detection from (a) detections csv file(s).

        Parameters
        ----------
        csv: Path | list[Path]
            Path of the detections csv file.
            If csv is a list, all detections from the multiple csv files
            are concatenated together.

        Returns
        -------
        list[Self]:
            List of detections taken from the csv file(s).
        """
        if type(csv) is not list:
            csv = [csv]

        output = []
        for csv_file in csv:
            output += cls._from_csv(csv_file)

        return output
