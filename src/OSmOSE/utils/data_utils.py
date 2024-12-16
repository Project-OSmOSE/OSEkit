"""Util classes and functions for data objects."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from pandas import Timestamp


@dataclass
class Event:
    """Dataclass containing begin an end attributes.

    Classes that have a begin and an end should inherit from Event.
    """

    begin: Timestamp
    end: Timestamp


TEvent = TypeVar("TEvent", bound=Event)


def is_overlapping(
    event1: TEvent | Event,
    event2: TEvent | Event,
) -> bool:
    """Return True if the two events are overlapping, False otherwise.

    Events are objects that have begin and end Timestamp attributes.

    Parameters
    ----------
    event1: Event
        The first event.
    event2: Event
        The second event.

    Returns
    -------
    bool:
        True if the two events are overlapping, False otherwise.

    Examples
    --------
    >>> is_overlapping(Event(begin=Timestamp("2024-01-01 00:00:00"),end=Timestamp("2024-01-02 00:00:00")), Event(begin=Timestamp("2024-01-01 12:00:00"),end=Timestamp("2024-01-02 12:00:00")))
    True
    >>> is_overlapping(Event(begin=Timestamp("2024-01-01 00:00:00"),end=Timestamp("2024-01-02 00:00:00")), Event(begin=Timestamp("2024-01-02 00:00:00"),end=Timestamp("2024-01-02 12:00:00")))
    False

    """
    return event1.begin < event2.end and event1.end > event2.begin


def remove_overlaps(events: list[TEvent]) -> list[TEvent]:
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
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Item:
    ...     begin: Timestamp
    ...     end: Timestamp
    >>> items = [Item(begin=Timestamp("00:00:00"),end=Timestamp("00:00:15")), Item(begin=Timestamp("00:00:10"),end=Timestamp("00:00:20"))]
    >>> events[0].end == events[1].begin
    False
    >>> items = remove_overlaps(events)
    >>> events[0].end == events[1].begin
    True

    """
    events = sorted(
        [copy.copy(event) for event in events],
        key=lambda event: (event.begin, event.begin - event.end),
    )
    concatenated_events = []
    for event in events:
        concatenated_events.append(event)
        overlapping_events = [
            event2
            for event2 in events
            if event2 is not event and is_overlapping(event, event2)
        ]
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


def fill_gaps(events: list[TEvent], filling_class: type[TEvent]) -> list[TEvent]:
    """Return a list with empty events added in the gaps between items.

    The created empty events are instantiated from the class filling_class.

    Parameters
    ----------
    events: list[TEvent]
        List of events to fill.
    filling_class: type[TEvent]
        The class used for instantiating empty events in the gaps.

    Returns
    -------
    list[TEvent]:
        List of events with no gaps.

    Examples
    --------
    >>> events = [Event(begin = Timestamp("00:00:00"), end = Timestamp("00:00:10")), Event(begin = Timestamp("00:00:15"), end = Timestamp("00:00:25"))]
    >>> events = fill_gaps(events, Event)
    >>> [(event.begin.second, event.end.second) for event in events]
    [(0, 10), (10, 15), (15, 25)]

    """
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
                filling_class(begin=event.end, end=next_event.begin),
            )
    filled_event_list.append(events[-1])
    return filled_event_list
