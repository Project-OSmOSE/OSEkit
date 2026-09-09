"""The Detection class represents a detection made on APLOSE."""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import TransformedBbox
from pandas import Timedelta, Timestamp

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

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the frequency bounds.

        Returns
        -------
        dict:
            The serialized frequency bounds, formatted for APLOSE.

        """
        return {
            "min_frequency": self.min,
            "max_frequency": self.max,
        }


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

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the detector info.

        Returns
        -------
        dict:
            The serialized detector info, formatted for APLOSE.

        """
        return {
            "annotator": self.name,
            "annotator_expertise": self.expertise,
        }


@dataclass
class SignalParameters:
    """Class representing parameters of detection signal."""

    is_intensity_too_low: bool | None = None
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

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the signal parameters.

        Returns
        -------
        dict:
            The serialized signal parameters, formatted for APLOSE.

        """
        return {
            "signal_is_intensity_too_low": self.is_intensity_too_low,
            "signal_does_overlap_other_signals": self.does_overlap_other_signals,
            "signal_frequency_jumps": self.frequency_jumps,
            "signal_deterministic_chaos": self.has_deterministic_chaos,
            "signal_has_harmonics": self.has_harmonics,
            "signal_sidebands": self.has_sidebands,
            "signal_subharmonics": self.has_subharmonics,
            "signal_end_frequency": self.max_frequency,
            "signal_start_frequency": self.min_frequency,
            "signal_relative_max_frequency_count": self.nb_relative_maxes,
            "signal_relative_min_frequency_count": self.nb_relative_mins,
            "signal_steps_count": self.nb_steps,
            "signal_trend": self.trend,
        }


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

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the confidence indicator.

        Returns
        -------
        dict:
            The serialized confidence indicator, formatted for APLOSE.

        """
        return {
            "confidence_indicator_label": self.label,
            "confidence_indicator_level": f"{self.level}/{self.maximum_level}",
        }


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

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the detection metadata.

        Returns
        -------
        dict:
            The serialized detection metadata, formatted for APLOSE.

        """
        return {
            "project": self.project,
            "filename": self.filename,
            "annotation_id": self.detection_id,
            "is_update_of_id": self.base_id,
            "comments": self.comments,
            "created_at_phase": self.phase,
        }


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

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the verification.

        Returns
        -------
        dict:
            The serialized verification, formatted for APLOSE.

        """
        return {
            self.verificator: self.is_validated,
        }


class Label:
    """Class that represents a label of a detection.

    It contains helper methods to plot the label next to the
    detection rectangle using pyplot.
    """

    def __init__(
        self,
        text: str,
        anchor: Literal[
            "top_left",
            "top_right",
            "bottom_right",
            "bottom_left",
        ] = "top_left",
        color: str = "white",
        text_color: str = "black",
        *,
        inner_text: bool = False,
        fill: bool = True,
        text_kwargs: dict | None = None,
        background_kwargs: dict | None = None,
    ) -> None:
        """Initialize the label object.

        Parameters
        ----------
        text: str
            Text of the label.
        anchor: Literal["top_left", "top_right", "bottom_right", "bottom_left"]
            Anchor of the label relative to the detection rectangle.
        color: str
            Color of the label rectangle.
        text_color: str
            Color of the label text.
        inner_text: bool
            If ``True``, the label rectangle is plotted inside the detection rectangle.
        fill: bool
            If ``True``, the label rectangle is plotted as a fill.
        text_kwargs: dict|None
            Additional kwargs to pass to the ``Text``.
        background_kwargs: dict|None
            Additional kwargs to pass to the background ``Rectangle``.

        """
        self.text = text
        self.anchor = anchor
        self.color = color
        self.text_color = text_color
        self.inner_text = inner_text
        self.fill = fill
        self.text_kwargs = text_kwargs or {}
        self.background_kwargs = background_kwargs or {}

    def get_text_size(self, ax: Axes) -> tuple[Timedelta, float]:
        """Return the width and height of the label text.

        The size is given in data coordinates.

        Parameters
        ----------
        ax: Axes
            Axes in which the text is drawn.

        Returns
        -------
        tuple[Timedelta, float]
            The width and height of the label text, in the
            data coordinates of the ``Axes`` object.

        """
        # We add a Text object with the given text to the Axes
        # to measure its size, then remove it
        text = Text(text=self.text, **self.text_kwargs)
        ax.add_artist(text)
        renderer = ax.get_figure().canvas.get_renderer()
        text_bbox = text.get_window_extent(renderer=renderer)  # display coordinates
        text.remove()

        # Conversion of the bbox in data units
        text_bbox = TransformedBbox(bbox=text_bbox, transform=ax.transData.inverted())
        return Timedelta(days=text_bbox.width), text_bbox.height

    def get_coordinates(
        self,
        ax: Axes,
        labelled_rect: Rectangle,
    ) -> tuple[float, float]:
        """Return the coordinates of the bottom left point of the label.

        The coordinates are given in data coordinates.

        Parameters
        ----------
        ax: Axes
            Axes in which the text is drawn.
        labelled_rect: Rectangle
            Rectangle that is labelled by the label.

        Returns
        -------
        tuple[float, float]:
            Coordinates of the bottom left point of the label, in
            data coordinates.

        """
        x0, y0 = labelled_rect.xy
        x1 = x0 + labelled_rect.get_width()
        y1 = y0 + labelled_rect.get_height()

        label_width, label_height = self.get_text_size(ax=ax)

        vertical_anchor, horizontal_anchor = self.anchor.split("_", maxsplit=1)

        label_x = x0 if horizontal_anchor == "left" else (x1 - label_width)
        if self.inner_text:
            label_y = y0 if vertical_anchor == "bottom" else y1 - label_height
        else:
            label_y = y0 - label_height if vertical_anchor == "bottom" else y1
        return label_x, label_y

    def get_rectangle(self, ax: Axes, labelled_rect: Rectangle) -> Rectangle:
        """Return the background rectangle of the label.

        Parameters
        ----------
        ax: Axes
            Axes in which the label is drawn.
        labelled_rect: Rectangle
            Rectangle that is labelled by the label.

        Returns
        -------
        Rectangle
            Background rectangle of the label

        """
        xy = self.get_coordinates(ax=ax, labelled_rect=labelled_rect)
        width, height = self.get_text_size(ax=ax)
        return Rectangle(
            xy=xy,
            height=height,
            width=width,
            color=self.color,
            fill=self.fill,
            **self.background_kwargs,
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
                does_overlap_other_signals=row.get("signal_does_overlap_other_signals"),
                frequency_jumps=row.get("signal_frequency_jumps"),
                has_deterministic_chaos=row.get("signal_deterministic_chaos"),
                has_harmonics=row.get("signal_has_harmonics"),
                has_sidebands=row.get("signal_sidebands"),
                has_subharmonics=row.get("signal_subharmonics"),
                is_intensity_too_low=row.get("signal_is_intensity_too_low"),
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
            signal_quantity=signal_quantity,
            signal_parameters=signal_parameters,
            verifications=verifications,
        )

    def to_dict(self) -> dict:
        """Return a serialized dictionary representation of the detection.

        Returns
        -------
        dict:
            The serialized detection, formatted for APLOSE.

        """
        return (
            {
                "annotation": self.label,
                "start_datetime": self.begin,
                "end_datetime": self.end,
                "type": self.type,
                "signal_quantity": self.signal_quantity,
            }
            | (self.metadata.to_dict() if self.metadata is not None else {})
            | (self.detector_info.to_dict() if self.detector_info is not None else {})
            | self.frequency_bounds.to_dict()
            | (
                self.confidence_indicator.to_dict()
                if self.confidence_indicator is not None
                else {}
            )
            | (
                self.signal_parameters.to_dict()
                if self.signal_parameters is not None
                else {}
            )
        ) | {
            verificator: verification
            for kvp in self.verifications
            for verificator, verification in kvp.to_dict().items()
        }

    def plot(
        self,
        ax: Axes,
        *,
        plot_label: bool,
        detection_rect_kwargs: dict | None = None,
        label_kwargs: dict | None = None,
    ) -> None:
        """Plot the detection on the given ``Axes``.

        Parameters
        ----------
        ax: Axes
            Axes in which to plot the detection box.
        plot_label: bool
            Whether or not to add the label in the detection plot.
        detection_rect_kwargs: dict|None
            Additional kwargs to pass to the detection rectangle.
        label_kwargs: dict|None
            Additional kwargs to pass to the ``Label`` object.

        """
        detection_rect_kwargs = detection_rect_kwargs or {}
        label_kwargs = label_kwargs or {}

        detection_rectangle = self.to_rectangle(**detection_rect_kwargs)
        ax.add_patch(detection_rectangle)

        if not plot_label:
            return

        # Default color is rectangle color
        if "color" in detection_rect_kwargs and "color" not in label_kwargs:
            label_kwargs["color"] = detection_rect_kwargs["color"]

        label = Label(
            text=self.label,
            **label_kwargs,
        )
        label_rectangle = label.get_rectangle(ax=ax, labelled_rect=detection_rectangle)
        ax.add_patch(p=label_rectangle)
        ax.annotate(
            text=label.text,
            xy=label.get_coordinates(ax=ax, labelled_rect=detection_rectangle),
            **label.text_kwargs,
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
    def _from_csv(cls, csv: Path, **kwargs: Any) -> list[Self]:
        records = (
            pd.read_csv(filepath_or_buffer=csv, **kwargs)
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
    def from_csv(cls, csv: Path | list[Path], **kwargs: Any) -> list[Self]:
        """Deserialize a list of Detection from (a) detections csv file(s).

        Parameters
        ----------
        csv: Path | list[Path]
            Path of the detections csv file.
            If csv is a list, all detections from the multiple csv files
            are concatenated together.
        **kwargs: Any
            Additional keyword arguments passed to the ``pandas.read_csv()`` method.

        Returns
        -------
        list[Self]:
            List of detections taken from the csv file(s).

        """
        if type(csv) is not list:
            csv = [csv]

        output = []
        for csv_file in csv:
            output += cls._from_csv(csv_file, **kwargs)

        return output
