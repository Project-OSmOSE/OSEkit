from __future__ import annotations

import pytest
from pandas import Timestamp

from OSmOSE.utils.data_utils import Event, fill_gaps, is_overlapping, remove_overlaps


@pytest.mark.parametrize(
    ("event1", "event2", "expected"),
    [
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            True,
            id="same_event",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            True,
            id="overlapping_events",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            True,
            id="overlapping_events_reversed",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-01 12:01:00"),
            ),
            True,
            id="embedded_events",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 0:00:00"),
                end=Timestamp("2024-01-01 12:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-02 00:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            False,
            id="non_overlapping_events",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-02 0:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-01 12:00:00"),
            ),
            False,
            id="non_overlapping_events_reversed",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 0:00:00"),
                end=Timestamp("2024-01-01 12:00:00"),
            ),
            Event(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            False,
            id="border_sharing_isnt_overlapping",
        ),
    ],
)
def test_overlapping_events(
    event1: Event,
    event2: Event,
    expected: bool,
) -> None:
    assert is_overlapping(event1, event2) == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            [Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            id="only_one_event_is_unchanged",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            ],
            [Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            id="doubled_event_is_removed",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:15")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            id="overlapping_event_is_truncated",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:15")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:15")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            id="longest_event_is_prioritized",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:25")),
                Event(begin=Timestamp("00:00:0"), end=Timestamp("00:00:15")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:35")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:35")),
            ],
            id="events_are_reordered",
        ),
    ],
)
def test_remove_overlaps(events: list[Event], expected: list[Event]) -> None:
    cleaned_events = remove_overlaps(events)
    assert cleaned_events == expected


@pytest.mark.parametrize(
    ("events", "expected"),
    [
        pytest.param(
            [Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            [Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            id="only_one_event_is_unchanged",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            id="consecutive_events_are_unchanged",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
            ],
            id="one_gap_is_filled",
        ),
        pytest.param(
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
                Event(begin=Timestamp("00:00:35"), end=Timestamp("00:00:45")),
                Event(begin=Timestamp("00:01:00"), end=Timestamp("00:02:00")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
                Event(begin=Timestamp("00:00:30"), end=Timestamp("00:00:35")),
                Event(begin=Timestamp("00:00:35"), end=Timestamp("00:00:45")),
                Event(begin=Timestamp("00:00:45"), end=Timestamp("00:01:00")),
                Event(begin=Timestamp("00:01:00"), end=Timestamp("00:02:00")),
            ],
            id="multiple_gaps_are_filled",
        ),
    ],
)
def test_fill_event_gaps(events: list[Event], expected: list[Event]) -> None:
    filled_events = fill_gaps(events, Event)
    assert filled_events == expected
