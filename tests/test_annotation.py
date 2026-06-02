from contextlib import AbstractContextManager, nullcontext

import pytest

from osekit.core.annotation import (
    AnnotatorInfo,
    FrequencyBounds,
)


@pytest.mark.parametrize(
    ("min_frequency", "max_frequency", "expectation"),
    [
        pytest.param(
            0,
            1000,
            nullcontext(1000),
            id="box_from_bottom",
        ),
        pytest.param(
            300,
            1000,
            nullcontext(700),
            id="box_bandwidth_from_higher_than_0",
        ),
        pytest.param(
            -10,
            1000,
            pytest.raises(ValueError, match=r"Min frequency.*-10"),
            id="negative_min_frequency_raises",
        ),
        pytest.param(
            0,
            -5,
            pytest.raises(ValueError, match=r"Max frequency.*-5"),
            id="negative_max_frequency_raises",
        ),
        pytest.param(
            80,
            50,
            pytest.raises(
                ValueError,
                match=r"Max frequency.*greater.*min frequency.*\(80,50\)",
            ),
            id="min_greater_than_max_raises",
        ),
        pytest.param(
            -20,
            -30,
            pytest.raises(
                ValueError,
                match=r"(?s)"  # Activates the DOTALL mode: includes \n in regex .*
                r"(?=.*Min frequency.*got -20)"
                r"(?=.*Max frequency.*got -30)"
                r"(?=.*Max frequency.*greater.*min frequency.*\(-20,-30\))",
            ),
            id="errors_concatenation",
        ),
    ],
)
def test_frequency_bounds(
    min_frequency: int,
    max_frequency: int,
    expectation: AbstractContextManager,
) -> None:
    with expectation as e:
        frequency_bounds = FrequencyBounds(min=min_frequency, max=max_frequency)
        assert frequency_bounds.bandwidth == e


def test_annotator_info() -> None:
    annotators = [
        AnnotatorInfo(annotator="ruby", annotator_expertise="NOVICE"),
        AnnotatorInfo(annotator="ruby", annotator_expertise="NOVICE"),
        AnnotatorInfo(annotator="haunt", annotator_expertise="EXPERT"),
        AnnotatorInfo(annotator="haunt", annotator_expertise="EXPERT"),
        AnnotatorInfo(annotator="nevada", annotator_expertise="EXPERT"),
        AnnotatorInfo(annotator="nevada", annotator_expertise="EXPERT"),
        AnnotatorInfo(annotator="haunt", annotator_expertise=None),
    ]

    nb_unique_annotators = 4

    assert sum(1 for _ in set(annotators)) == nb_unique_annotators
