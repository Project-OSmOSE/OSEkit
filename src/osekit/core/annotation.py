"""The Annotation class represents an annotation made on APLOSE."""

from pandas import Timestamp

from osekit.core.event import Event


class Annotation(Event):
    """Class represents an annotation made on APLOSE."""

    def __init__(  # noqa: PLR0913
        self,
        project: str,
        begin: Timestamp,
        end: Timestamp,
        min_frequency: int,
        max_frequency: int,
        annotation: str,
        annotator: str,
        confidence_indicator_label: str,
        confidence_indicator_level: str,
        *,
        is_box: bool,
    ) -> None:
        """Initialize an Annotation object.

        Parameters
        ----------
        project: str
            Name of the project in which the annotation was made.
        begin: Timestamp
            Begin timestamp of the annotation.
        end: Timestamp
            End timestamp of the annotation.
        min_frequency: int
            Minimum frequency of the annotation.
        max_frequency: int
            Maximum frequency of the annotation.
        annotation: str
            Label of the annotation.
        annotator: str
            Name of the annotator or detector.
        confidence_indicator_label: str
            Name of the level of confidence.
        confidence_indicator_level: str
            Level of confidence relative to the maximum level available.
            Should be formatted as ``n/m``, where ``n`` is the level of confidence
            of the annotation and ``m`` is the maximum level available in the project.
        is_box: bool
            If ``True``, the annotation is a box.
            If ``False``, the annotation is a weak annotation.

        """
        super().__init__(begin=begin, end=end)
        self.project = project
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency
        self.annotation = annotation
        self.annotator = annotator
        self.confidence_indicator_label = confidence_indicator_label
        self.confidence_indicator_level = confidence_indicator_level
        self.is_box = is_box
