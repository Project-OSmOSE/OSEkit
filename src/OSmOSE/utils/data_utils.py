import copy
from dataclasses import dataclass
from typing import Protocol

from pandas import Timestamp


class Event(Protocol):
    begin: Timestamp
    end: Timestamp


@dataclass
class EventClass:
    begin: Timestamp
    end: Timestamp


def is_overlapping(
    event1: Event,
    event2: Event,
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
    >>> is_overlapping(EventClass(begin=Timestamp("2024-01-01 00:00:00"),end=Timestamp("2024-01-02 00:00:00")), EventClass(begin=Timestamp("2024-01-01 12:00:00"),end=Timestamp("2024-01-02 12:00:00")))
    True
    >>> is_overlapping(EventClass(begin=Timestamp("2024-01-01 00:00:00"),end=Timestamp("2024-01-02 00:00:00")), EventClass(begin=Timestamp("2024-01-02 00:00:00"),end=Timestamp("2024-01-02 12:00:00")))
    False

    """
    return event1.begin < event2.end and event1.end > event2.begin


def remove_overlaps(items: list) -> list:
    """Resolve overlaps between objects that have begin and end attributes.

    If two objects overlap within the sequence
    (that is if one object begins before the end of another),
    the earliest object's end is set to the begin of the latest object.
    If multiple objects overlap with one earlier object, only one is chosen as next.
    The chosen next object is the one that ends the latest.

    Parameters
    ----------
    items: list
        List of objects to concatenate.

    Returns
    -------
    list:
        The list of objects with no overlap.

    Examples
    --------
    >>> from dataclasses import dataclass
    >>> @dataclass
    ... class Item:
    ...     begin: Timestamp
    ...     end: Timestamp
    >>> items = [Item(begin=Timestamp("00:00:00"),end=Timestamp("00:00:15")), Item(begin=Timestamp("00:00:10"),end=Timestamp("00:00:20"))]
    >>> items[0].end == items[1].begin
    False
    >>> items = remove_overlaps(items)
    >>> items[0].end == items[1].begin
    True

    """
    items = sorted(
        [copy.copy(item) for item in items],
        key=lambda item: (item.begin, item.begin - item.end),
    )
    concatenated_items = []
    for item in items:
        concatenated_items.append(item)
        overlapping_items = [
            item2
            for item2 in items
            if item2 is not item and is_overlapping(item, item2)
        ]
        if not overlapping_items:
            continue
        kept_overlapping_item = max(overlapping_items, key=lambda item: item.end)
        if kept_overlapping_item.end > item.end:
            item.end = kept_overlapping_item.begin
        else:
            kept_overlapping_item = None
        for dismissed_item in (
            item2 for item2 in overlapping_items if item2 is not kept_overlapping_item
        ):
            items.remove(dismissed_item)
    return concatenated_items
