"""Event class."""

from __future__ import annotations

import bisect
import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from pandas import Timedelta, Timestamp


@dataclass
class Event:
    """Events are bounded between begin an end attributes.

    Classes that have a beginning and an end should inherit from Event.
    """

    _begin: Timestamp = field(init=False, repr=False, compare=True)
    _end: Timestamp = field(init=False, repr=False, compare=True)

    def __init__(
        self,
        begin: Timestamp,
        end: Timestamp,
    ) -> None:
        """Initialize an Event instance with a beginning and an end."""
        self.begin = begin
        self.end = end

    @property
    def begin(self) -> Timestamp:
        """Beginning of the event."""
        return self._begin

    @begin.setter
    def begin(self, value: Timestamp) -> None:
        if hasattr(self, "_end") and value >= self._end:
            msg = f"Invalid Event: `end` ({self._end}) must be greater than `begin` ({value})."  # noqa: E501
            raise ValueError(msg)
        self._begin = value

    @property
    def end(self) -> Timestamp:
        """End of the event."""
        return self._end

    @end.setter
    def end(self, value: Timestamp) -> None:
        if hasattr(self, "_begin") and value <= self._begin:
            msg = f"Invalid Event: `end` ({value}) must be greater than `begin` ({self._begin})."  # noqa: E501
            raise ValueError(msg)
        self._end = value

    @property
    def duration(self) -> Timedelta:
        """Duration of the event."""
        return self.end - self.begin

    def __repr__(self) -> str:
        """Overwrite repr."""
        return f"{self.begin} - {self.end}"

    def overlaps(self, other: type[Event] | Event) -> bool:
        """Return ``True`` if the other event shares time with the current event.

        Parameters
        ----------
        other: type[Event] | Event
            The other event.

        Returns
        -------
        bool:
            ``True`` if the two events are overlapping, ``False`` otherwise.
            Two events overlapping means that any timestamp is shared between the two events.

        Examples
        --------
        >>> from pandas import Timestamp
        >>> Event(begin=Timestamp("2024-01-01 00:00:00"),end=Timestamp("2024-01-02 00:00:00")).overlaps(Event(begin=Timestamp("2024-01-01 12:00:00"),end=Timestamp("2024-01-02 12:00:00")))
        True
        >>> Event(begin=Timestamp("2024-01-01 00:00:00"),end=Timestamp("2024-01-02 00:00:00")).overlaps(Event(begin=Timestamp("2024-01-02 00:00:00"),end=Timestamp("2024-01-02 12:00:00")))
        False

        """  # noqa: E501
        return self.begin < other.end and self.end > other.begin

    def get_overlapping_events(self, events: list[TEvent]) -> list[TEvent]:
        """Return a list of events that overlap with the current event.

        The events input list must be sorted by ``begin`` and ``end`` Timestamps.

        Parameters
        ----------
        events: list[TEvent]
            The list of events to be filtered by overlap.
            It must be sorted by ``begin`` and ``end`` Timestamps.

        Returns
        -------
        list[TEvent]:
            The events from the events input list that overlap with the current event.

        """
        output = []
        start_index = bisect.bisect_left(
            events,
            self.begin,
            key=lambda event: event.end,
        )
        for i in range(start_index, len(events)):
            if events[i] is self:
                continue
            if self.overlaps(events[i]):
                output.append(events[i])
                continue
            if events[i].begin >= self.end:
                break
        return output

    @classmethod
    def remove_overlaps(cls, events: list[TEvent]) -> list[TEvent]:
        """Resolve overlaps between events.

        If two events overlap within the whole events collection
        (that is if one event begins before the end of another event),
        the earliest event's end is set to the begin of the latest object.
        If multiple events overlap with one earlier event, only one is chosen as next.
        The chosen next event is the one that ends the latest.

        Parameters
        ----------
        events: list
            List of events in which to remove the overlaps.

        Returns
        -------
        list:
            The list of events with no overlap.

        Examples
        --------
        >>> from pandas import Timestamp
        >>> events = [Event(begin=Timestamp("00:00:00"),end=Timestamp("00:00:15")), Event(begin=Timestamp("00:00:10"),end=Timestamp("00:00:20"))]
        >>> events[0].end == events[1].begin
        False
        >>> events = Event.remove_overlaps(events)
        >>> events[0].end == events[1].begin
        True

        """  # noqa: E501
        events = sorted(
            events,
            key=lambda event: (event.begin, -1 * event.duration),
        )
        concatenated_events = []
        for event in events:
            concatenated_events.append(event)
            overlapping_events = event.get_overlapping_events(events)
            if not overlapping_events:
                continue
            kept_overlapping_event = max(overlapping_events, key=lambda item: item.end)
            if kept_overlapping_event.end > event.end:
                event.end = kept_overlapping_event.begin
            else:
                kept_overlapping_event = None
            for dismissed_event in (
                event2
                for event2 in overlapping_events
                if event2 is not kept_overlapping_event
            ):
                events.remove(dismissed_event)
        return concatenated_events

    @classmethod
    def fill_gaps(
        cls,
        events: list[TEvent],
        filling_class: type[TEvent],
        **kwargs,  # noqa: ANN003
    ) -> list[TEvent]:
        """Return a list with empty events added in the gaps between items.

        The created empty events are instantiated from the ``filling_class`` class.

        Parameters
        ----------
        events: list[TEvent]
            List of events to fill.
        filling_class: type[TEvent]
            The class used for instantiating empty events in the gaps.
        kwargs
            Additional parameters to pass to the filling instance constructor.

        Returns
        -------
        list[TEvent]:
            List of events with no gaps.

        Examples
        --------
        >>> from pandas import Timestamp
        >>> events = [Event(begin = Timestamp("00:00:00"), end = Timestamp("00:00:10")), Event(begin = Timestamp("00:00:15"), end = Timestamp("00:00:25"))]
        >>> events = Event.fill_gaps(events, Event)
        >>> [(event.begin.second, event.end.second) for event in events]
        [(0, 10), (10, 15), (15, 25)]

        """  # noqa: E501
        events = sorted(
            [copy.copy(event) for event in events],
            key=lambda event: event.begin,
        )
        filled_event_list = []
        for index, event in enumerate(events[:-1]):
            next_event = events[index + 1]
            filled_event_list.append(event)
            if next_event.begin > event.end:
                filled_event_list.append(
                    filling_class(begin=event.end, end=next_event.begin, **kwargs),
                )
        filled_event_list.append(events[-1])
        return filled_event_list


TEvent = TypeVar("TEvent", bound=Event)
