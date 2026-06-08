from contextlib import AbstractContextManager, nullcontext
from pathlib import Path

import numpy as np
import pytest
from pandas import Timestamp

from osekit.core.annotation import (
    Annotation,
    AnnotationMetaData,
    AnnotatorInfo,
    ConfidenceIndicator,
    FrequencyBounds,
    SignalParameters,
    Verification,
)


@pytest.fixture
def sample_annotation() -> Annotation:
    return Annotation(
        metadata=AnnotationMetaData(
            annotation_id=35173,
            base_id=None,
            comments="He's a sneaky, sneaky dog friend",
            filename="its_teasy",
            phase="ANNOTATION",
            project="mockasin",
        ),
        begin=Timestamp("2013-11-05 00:00:00"),
        end=Timestamp("2013-11-05 00:00:10"),
        frequency_bounds=FrequencyBounds(
            min=1_000,
            max=3_000,
        ),
        label="Connan",
        annotator_info=AnnotatorInfo(
            annotator="Mockasin",
            annotator_expertise="EXPERT",
        ),
        annotation_type="BOX",
        confidence_indicator=ConfidenceIndicator(
            label="Sure",
            level=2,
            maximum_level=2,
        ),
        signal_quantity="SINGLE",
        signal_parameters=SignalParameters(
            does_overlap_other_signals=False,
            frequency_jumps=True,
            has_deterministic_chaos=True,
            has_harmonics=True,
            has_sidebands=True,
            has_subharmonics=False,
            is_itensity_too_low=False,
            max_frequency=2_800,
            min_frequency=1_300,
            nb_relative_maxes=2,
            nb_relative_mins=3,
            nb_steps=4,
            trend="MODULATED",
        ),
        verifications={
            Verification(
                verificator="soft_hair",
                is_validated=True,
            ),
        },
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


@pytest.mark.parametrize(
    ("label", "level", "max_level", "expectation"),
    [
        pytest.param(
            "Sure",
            1,
            1,
            nullcontext(),
            id="max_level_is_ok",
        ),
        pytest.param(
            "Not sure",
            0,
            1,
            nullcontext(),
            id="level_0_is_ok",
        ),
        pytest.param(
            "Moderate",
            1,
            2,
            nullcontext(),
            id="between_0_and_max_is_ok",
        ),
        pytest.param(
            "Moderate",
            3,
            2,
            pytest.raises(ValueError, match=r"level 3.*higher.*maximum level 2"),
            id="higher_than_max_raises",
        ),
    ],
)
def test_confidence_indicator_value_check(
    label: str,
    level: int,
    max_level: int,
    expectation: AbstractContextManager,
) -> None:
    with expectation:
        ConfidenceIndicator(
            label=label,
            level=level,
            maximum_level=max_level,
        )


@pytest.mark.parametrize(
    ("label", "relative_level_string", "expectation"),
    [
        pytest.param(
            "cool",
            "1/6",
            nullcontext(
                ConfidenceIndicator(
                    label="cool",
                    level=1,
                    maximum_level=6,
                ),
            ),
            id="correct_levels",
        ),
        pytest.param(
            "cool",
            "4/2",
            pytest.raises(ValueError, match=r"level 4.*higher.*maximum level 2"),
            id="incorrect_levels_should_raise",
        ),
    ],
)
def test_confidence_indicator_from_relative_level_string(
    label: str,
    relative_level_string: str,
    expectation: AbstractContextManager,
) -> None:
    with expectation as e:
        ci = ConfidenceIndicator.from_relative_level_string(
            label=label,
            relative_level_string=relative_level_string,
        )

        assert ci.label == e.label
        assert ci.level == e.level
        assert ci.maximum_level == e.maximum_level


def test_annotations_from_csv() -> None:
    annotations = Annotation.from_csv(
        csv=Path(__file__).parent / "_static" / "aplose_result.csv",
    )

    # All records should be loaded
    assert len(annotations) == 8
    assert all(a.metadata.project == "great_tit" for a in annotations)

    # Two distinct annotated files
    filenames = {a.metadata.filename for a in annotations}
    assert filenames == {"990694", "994410"}

    # Types
    types = {a.type for a in annotations}
    assert types == {"WEAK", "BOX"}

    # Phases
    phases = {a.metadata.phase for a in annotations}
    assert phases == {"ANNOTATION", "VERIFICATION"}

    # Single signal parameters
    single = next(a for a in annotations if a.metadata.annotation_id == 586657)
    assert single.signal_quantity == "SINGLE"
    assert single.signal_parameters is not None
    assert not single.signal_parameters.is_itensity_too_low
    assert not single.signal_parameters.does_overlap_other_signals
    assert single.signal_parameters.min_frequency == 12000
    assert single.signal_parameters.max_frequency == 13000
    assert single.signal_parameters.nb_relative_mins == 3
    assert single.signal_parameters.nb_relative_maxes == 2
    assert single.signal_parameters.nb_steps == 4
    assert single.signal_parameters.trend == "MOD"
    assert single.signal_parameters.frequency_jumps
    assert single.signal_parameters.has_harmonics
    assert single.signal_parameters.has_sidebands
    assert not single.signal_parameters.has_subharmonics
    assert single.signal_parameters.has_deterministic_chaos

    # Multiple signal quantity: parameters should be None
    multiple = next(a for a in annotations if a.metadata.annotation_id == 586654)
    assert multiple.signal_quantity == "MULTIPLE"
    assert multiple.signal_parameters is None

    # Annotation update
    update = next(a for a in annotations if a.metadata.annotation_id == 586669)
    assert update.metadata.base_id == 586655

    # Annotation without base
    base = next(a for a in annotations if a.metadata.annotation_id == 586655)
    assert base.metadata.base_id is None

    # Annotator parsing
    annotators = {
        AnnotatorInfo(annotator="vashti", annotator_expertise="NOVICE"),
        AnnotatorInfo(annotator="heartleap", annotator_expertise=None),
        AnnotatorInfo(annotator="bunyan", annotator_expertise="EXPERT"),
        AnnotatorInfo(annotator="lookaftering", annotator_expertise="EXPERT"),
    }
    assert np.array_equal(
        annotators,
        {a.annotator_info for a in annotations},
    )

    # Verification parsing
    verificated = next(a for a in annotations if a.metadata.annotation_id == 586654)
    verification = {
        Verification(
            verificator="lookaftering",
            is_validated=True,
        ),
        Verification(
            verificator="bunyan",
            is_validated=False,
        ),
    }
    assert np.array_equal(verification, verificated.verifications)

    # Repr should be the annotation ID
    annotation = annotations[0]
    assert str(annotation) == str(annotation.metadata.annotation_id)
