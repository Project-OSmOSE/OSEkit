from __future__ import annotations

import pytest
from pandas import Timestamp

from OSmOSE.data.base_item import BaseItem


@pytest.mark.parametrize(
    ("item_list", "expected"),
    [
        pytest.param(
            [BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            [BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            id="only_one_item_is_unchanged",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
            ],
            [BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            id="doubled_item_is_removed",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:15")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            id="overlapping_item_is_truncated",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:15")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:15")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            id="longest_item_is_prioritized",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:25")),
                BaseItem(begin=Timestamp("00:00:0"), end=Timestamp("00:00:15")),
                BaseItem(begin=Timestamp("00:00:20"), end=Timestamp("00:00:35")),
            ],
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
                BaseItem(begin=Timestamp("00:00:20"), end=Timestamp("00:00:35")),
            ],
            id="items_are_reordered",
        ),
    ],
)
def test_remove_overlaps(item_list: list[BaseItem], expected: list[BaseItem]) -> None:
    cleaned_items = BaseItem.remove_overlaps(item_list)
    assert cleaned_items == expected


@pytest.mark.parametrize(
    ("item_list", "expected"),
    [
        pytest.param(
            [BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            [BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10"))],
            id="only_one_item_is_unchanged",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
            ],
            id="consecutive_items_are_unchanged",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
            ],
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
                BaseItem(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
            ],
            id="one_gap_is_filled",
        ),
        pytest.param(
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
                BaseItem(begin=Timestamp("00:00:35"), end=Timestamp("00:00:45")),
                BaseItem(begin=Timestamp("00:01:00"), end=Timestamp("00:02:00")),
            ],
            [
                BaseItem(begin=Timestamp("00:00:00"), end=Timestamp("00:00:10")),
                BaseItem(begin=Timestamp("00:00:10"), end=Timestamp("00:00:20")),
                BaseItem(begin=Timestamp("00:00:20"), end=Timestamp("00:00:30")),
                BaseItem(begin=Timestamp("00:00:30"), end=Timestamp("00:00:35")),
                BaseItem(begin=Timestamp("00:00:35"), end=Timestamp("00:00:45")),
                BaseItem(begin=Timestamp("00:00:45"), end=Timestamp("00:01:00")),
                BaseItem(begin=Timestamp("00:01:00"), end=Timestamp("00:02:00")),
            ],
            id="multiple_gaps_are_filled",
        ),
    ],
)
def test_fill_item_gaps(item_list: list[BaseItem], expected: list[BaseItem]) -> None:
    filled_items = BaseItem.fill_gaps(item_list)
    assert filled_items == expected
