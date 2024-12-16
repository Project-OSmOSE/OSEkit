import pytest
from pandas import Timestamp

from OSmOSE.utils.data_utils import EventClass, is_overlapping


@pytest.mark.parametrize(
    ("event1", "event2", "expected"),
    [
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            True,
            id="same_event",
        ),
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            True,
            id="overlapping_events",
        ),
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            True,
            id="overlapping_events_reversed",
        ),
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-01 12:01:00"),
            ),
            True,
            id="embedded_events",
        ),
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-01 0:00:00"),
                end=Timestamp("2024-01-01 12:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-02 00:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            False,
            id="non_overlapping_events",
        ),
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-02 0:00:00"),
                end=Timestamp("2024-01-02 12:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-01 00:00:00"),
                end=Timestamp("2024-01-01 12:00:00"),
            ),
            False,
            id="non_overlapping_events_reversed",
        ),
        pytest.param(
            EventClass(
                begin=Timestamp("2024-01-01 0:00:00"),
                end=Timestamp("2024-01-01 12:00:00"),
            ),
            EventClass(
                begin=Timestamp("2024-01-01 12:00:00"),
                end=Timestamp("2024-01-02 00:00:00"),
            ),
            False,
            id="border_sharing_isnt_overlapping",
        ),
    ],
)
def test_overlapping_events(
    event1: EventClass,
    event2: EventClass,
    expected: bool,
) -> None:
    assert is_overlapping(event1, event2) == expected
