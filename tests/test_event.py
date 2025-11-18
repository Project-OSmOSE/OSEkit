from __future__ import annotations

from contextlib import nullcontext

import pytest
from pandas import Timestamp

from osekit.core_api.event import Event


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
    assert event1.overlaps(event2) is expected


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
    cleaned_events = Event.remove_overlaps(events)
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
    filled_events = Event.fill_gaps(events, Event)
    assert filled_events == expected


@pytest.mark.parametrize(
    ("event", "events", "expected"),
    [
        pytest.param(
            Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            [],
            [],
            id="no_event",
        ),
        pytest.param(
            Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            ],
            id="one_identical_event",
        ),
        pytest.param(
            Event(begin=Timestamp("00:00:05"), end=Timestamp("00:00:15")),
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            ],
            id="one_overlapping_from_start",
        ),
        pytest.param(
            Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            [
                Event(begin=Timestamp("00:00:05"), end=Timestamp("00:00:15")),
            ],
            [
                Event(begin=Timestamp("00:00:05"), end=Timestamp("00:00:15")),
            ],
            id="one_overlapping_from_end",
        ),
        pytest.param(
            Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            [
                Event(begin=Timestamp("00:00:15"), end=Timestamp("00:00:20")),
            ],
            [],
            id="one_not_overlapping",
        ),
        pytest.param(
            Event(begin=Timestamp("00:00:05"), end=Timestamp("00:00:10")),
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:05")),
                Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:15")),
            ],
            [],
            id="direct_following_is_not_overlapping",
        ),
        pytest.param(
            Event(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:05")),
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:07")),
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:12")),
                Event(begin=Timestamp("00:00:05"), end=Timestamp("00:00:10")),
                Event(begin=Timestamp("00:00:08"), end=Timestamp("00:00:15")),
                Event(begin=Timestamp("00:00:12"), end=Timestamp("00:00:18")),
                Event(begin=Timestamp("00:00:16"), end=Timestamp("00:00:24")),
                Event(begin=Timestamp("00:00:18"), end=Timestamp("00:00:30")),
                Event(begin=Timestamp("00:00:20"), end=Timestamp("00:00:25")),
                Event(begin=Timestamp("00:00:30"), end=Timestamp("00:00:35")),
            ],
            [
                Event(begin=Timestamp("00:00:00"), end=Timestamp("00:00:12")),
                Event(begin=Timestamp("00:00:08"), end=Timestamp("00:00:15")),
                Event(begin=Timestamp("00:00:12"), end=Timestamp("00:00:18")),
                Event(begin=Timestamp("00:00:16"), end=Timestamp("00:00:24")),
                Event(begin=Timestamp("00:00:18"), end=Timestamp("00:00:30")),
            ],
            id="full_mix",
        ),
    ],
)
def test_get_overlapping_events(
    event: Event,
    events: list[Event],
    expected: list[Event],
) -> None:
    events = sorted(events, key=lambda e: (e.begin, e.end))

    overlap_result = sorted(event.get_overlapping_events(events), key=lambda e: e.begin)
    expected_result = sorted(expected, key=lambda e: e.begin)

    assert all(
        result == expected
        for result, expected in zip(overlap_result, expected_result, strict=False)
    )

    assert len(overlap_result) == len(expected_result)

    # event instance that is in the events list should be excluded
    events.append(event)
    events = sorted(events, key=lambda e: (e.begin, e.end))
    overlap_result = sorted(event.get_overlapping_events(events), key=lambda e: e.begin)
    assert all(
        result == expected
        for result, expected in zip(overlap_result, expected_result, strict=False)
    )

    assert len(overlap_result) == len(expected_result)


@pytest.mark.parametrize(
    ("event", "updated_begin", "updated_end", "expected"),
    [
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            Timestamp("2024-01-01 12:00:00"),
            None,
            nullcontext(
                Event(
                    begin=Timestamp("2024-01-01 12:00:00"),
                    end=Timestamp("2024-01-02 00:00:00"),
                ),
            ),
            id="valid_begin",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            None,
            Timestamp("2024-01-02 12:00:00"),
            nullcontext(
                Event(
                    begin=Timestamp("2024-01-01 00:00:00"),
                    end=Timestamp("2024-01-02 12:00:00"),
                ),
            ),
            id="valid_end",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            Timestamp("2024-01-03 00:00:00"),
            None,
            pytest.raises(ValueError, match="`end`.*must be greater than `begin`.*"),
            id="invalid_begin_after_end",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            None,
            Timestamp("2023-12-31 23:59:59"),
            pytest.raises(ValueError, match="`end`.*must be greater than `begin`.*"),
            id="invalid_end_before_begin",
        ),
        pytest.param(
            Event(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-01 01:00:00"),
            ),
            Timestamp("2024-01-01 01:00:00"),
            None,
            pytest.raises(ValueError, match="`end`.*must be greater than `begin`.*"),
            id="begin_equals_end",
        ),
    ],
)
def test_event_begin_end_updates(
    event: Event,
    updated_begin: Timestamp | None,
    updated_end: Timestamp | None,
    expected: Event,
) -> None:
    def update_event(
        cool_event: Event,
        begin: Timestamp | None,
        end: Timestamp | None,
    ) -> Event:
        if begin:
            cool_event.begin = begin
        if end:
            cool_event.end = end
        return cool_event

    with expected as e:
        assert update_event(event, updated_begin, updated_end) == e


@pytest.mark.parametrize(
    ("begin", "end"),
    [
        pytest.param(
            Timestamp("2024-01-02 00:00:00"),
            Timestamp("2024-01-01 00:00:00"),
            id="begin_after_end",
        ),
        pytest.param(
            Timestamp("2024-01-01 00:00:00"),
            Timestamp("2024-01-01 00:00:00"),
            id="begin_equals_end",
        ),
    ],
)
def test_event_errors(begin: Timestamp, end: Timestamp) -> None:
    with pytest.raises(ValueError, match="`end`.*must be greater than `begin`.*") as e:
        assert Event(begin=begin, end=end) == e


def test_repr() -> None:
    assert (
        repr(
            Event(
                begin=Timestamp("1990-09-12 12:00:00"),
                end=Timestamp("1990-09-12 12:00:10"),
            ),
        )
        == "1990-09-12 12:00:00 - 1990-09-12 12:00:10"
    )
