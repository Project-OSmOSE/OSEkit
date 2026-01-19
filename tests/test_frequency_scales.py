import contextlib

import numpy as np
import pytest

from osekit.core_api.frequency_scale import Scale, ScalePart


@pytest.mark.parametrize(
    ("scale_part", "nb_points", "expected"),
    [
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=1.0,
                f_min=0,
                f_max=3,
            ),
            4,
            [0, 1, 2, 3],
            id="simple_case",
        ),
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=1.0,
                f_min=10,
                f_max=12,
            ),
            6,
            [10, 10, 11, 11, 12, 12],
            id="float_frequencies_are_rounded",
        ),
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=1.0,
                f_min=1.0,
                f_max=10.0,
                scale_type="log",
            ),
            10,
            [1, 1, 2, 2, 3, 4, 5, 6, 8, 10],
            id="logarithmic_scale",
        ),
    ],
)
def test_frequency_scale_part_get_frequencies(
    scale_part: ScalePart,
    nb_points: int,
    expected: list[int],
) -> None:
    assert list(scale_part.get_frequencies(nb_points)) == expected


@pytest.mark.parametrize(
    ("scale_part", "scale_length", "expected"),
    [
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=1.0,
                f_min=0,
                f_max=500,
            ),
            10,
            (0, 10),
            id="full_scale",
        ),
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=0.5,
                f_min=0,
                f_max=500,
            ),
            10,
            (0, 5),
            id="first_half",
        ),
        pytest.param(
            ScalePart(
                p_min=0.5,
                p_max=1.0,
                f_min=0,
                f_max=500,
            ),
            10,
            (5, 10),
            id="last_half",
        ),
        pytest.param(
            ScalePart(
                p_min=0.18,
                p_max=0.38,
                f_min=0,
                f_max=500,
            ),
            10,
            (1, 3),
            id="float_index_to_floor",
        ),
    ],
)
def test_frequency_scale_part_get_indexes(
    scale_part: ScalePart,
    scale_length: int,
    expected: list[int],
) -> None:
    assert scale_part.get_indexes(scale_length) == expected


@pytest.mark.parametrize(
    ("scale_part", "nb_points", "expected"),
    [
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=1.0,
                f_min=0,
                f_max=3,
            ),
            4,
            [0, 1, 2, 3],
            id="full_scale",
        ),
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=1.0,
                f_min=100,
                f_max=200,
            ),
            5,
            [100, 125, 150, 175, 200],
            id="full_scale_large_frequencies",
        ),
        pytest.param(
            ScalePart(
                p_min=0.0,
                p_max=0.5,
                f_min=100,
                f_max=200,
            ),
            5,
            [100, 200],
            id="odd_first_half",
        ),
        pytest.param(
            ScalePart(
                p_min=0.5,
                p_max=1.0,
                f_min=200,
                f_max=300,
            ),
            5,
            [200, 250, 300],
            id="odd_second_half",
        ),
        pytest.param(
            ScalePart(
                p_min=0,
                p_max=0.5,
                f_min=100,
                f_max=200,
                scale_type="log",
            ),
            10,
            [100, 119, 141, 168, 200],
            id="logarithmic_scale",
        ),
    ],
)
def test_frequency_scale_part_get_values(
    scale_part: ScalePart,
    nb_points: int,
    expected: list[int],
) -> None:
    assert np.allclose(scale_part.get_values(nb_points), expected, atol=0.5)


@pytest.mark.parametrize(
    ("scale", "scale_length", "expected"),
    [
        pytest.param(
            Scale(
                parts=[
                    ScalePart(
                        p_min=0,
                        p_max=1.0,
                        f_min=0,
                        f_max=3,
                    ),
                ],
            ),
            4,
            [0, 1, 2, 3],
            id="one_full_part",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(
                        p_min=0,
                        p_max=0.5,
                        f_min=0,
                        f_max=1,
                    ),
                    ScalePart(
                        p_min=0.5,
                        p_max=1.0,
                        f_min=2,
                        f_max=3,
                    ),
                ],
            ),
            4,
            [0, 1, 2, 3],
            id="even_length_cut_in_half",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(
                        p_min=0,
                        p_max=0.5,
                        f_min=0,
                        f_max=1,
                    ),
                    ScalePart(
                        p_min=0.5,
                        p_max=1.0,
                        f_min=2,
                        f_max=4,
                    ),
                ],
            ),
            5,
            [0, 1, 2, 3, 4],
            id="odd_length_cut_in_half",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(
                        p_min=0,
                        p_max=0.1,
                        f_min=100,
                        f_max=200,
                    ),
                    ScalePart(
                        p_min=0.1,
                        p_max=0.5,
                        f_min=500,
                        f_max=1_200,
                    ),
                    ScalePart(
                        p_min=0.5,
                        p_max=1.0,
                        f_min=3_100,
                        f_max=4_000,
                    ),
                ],
            ),
            20,
            [
                100,
                200,
                500,
                600,
                700,
                800,
                900,
                1_000,
                1_100,
                1_200,
                3_100,
                3_200,
                3_300,
                3_400,
                3_500,
                3_600,
                3_700,
                3_800,
                3_900,
                4_000,
            ],
            id="non_consecutive_parts",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(
                        p_min=0,
                        p_max=0.5,
                        f_min=1,
                        f_max=100,
                        scale_type="log",
                    ),
                    ScalePart(
                        p_min=0.5,
                        p_max=1.0,
                        f_min=1000,
                        f_max=5000,
                        scale_type="log",
                    ),
                ],
            ),
            10,
            [1, 3, 10, 32, 100, 1000, 1495, 2236, 3344, 5000],
            id="logarithmic_scale",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(
                        p_min=0,
                        p_max=0.5,
                        f_min=1,
                        f_max=100,
                        scale_type="log",
                    ),
                    ScalePart(
                        p_min=0.5,
                        p_max=1.0,
                        f_min=1000,
                        f_max=5000,
                        scale_type="lin",
                    ),
                ],
            ),
            10,
            [1, 3, 10, 32, 100, 1000, 2000, 3000, 4000, 5000],
            id="scale_type_mix",
        ),
    ],
)
def test_frequency_scale_map(
    scale: Scale,
    scale_length: int,
    expected: list[float],
) -> None:
    assert np.allclose(scale.map(scale_length), expected, atol=0.5)


@pytest.mark.parametrize(
    ("scale", "original_scale", "expected"),
    [
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=1.0, f_min=0, f_max=5.0),
                ],
            ),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0, 1, 2, 3, 4, 5],
            id="same_scale",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=0.5, f_min=0, f_max=2.0),
                    ScalePart(p_min=0.5, p_max=1.0, f_min=3.0, f_max=5.0),
                ],
            ),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0, 1, 2, 3, 4, 5],
            id="same_scale_in_two_parts",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=0.8, f_min=0, f_max=1.0),
                    ScalePart(p_min=0.8, p_max=1.0, f_min=8.0, f_max=9.0),
                ],
            ),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0, 0, 0, 0, 1, 1, 1, 1, 8, 9],
            id="two_different_parts",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=0.8, f_min=100, f_max=110.0),
                    ScalePart(p_min=0.8, p_max=1.0, f_min=180.0, f_max=190.0),
                ],
            ),
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
            [0, 0, 0, 0, 1, 1, 1, 1, 8, 9],
            id="different_frequencies_same_indexes",
        ),
    ],
)
def test_frequency_scale_mapped_indexes(
    scale: Scale,
    original_scale: list[float],
    expected: list[int],
) -> None:
    assert scale.get_mapped_indexes(original_scale=original_scale) == expected


@pytest.mark.parametrize(
    ("scale", "original_scale", "expected"),
    [
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=1.0, f_min=0, f_max=5.0),
                ],
            ),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            id="same_scale",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=0.5, f_min=0, f_max=2.0),
                    ScalePart(p_min=0.5, p_max=1.0, f_min=3.0, f_max=5.0),
                ],
            ),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            id="same_scale_in_two_parts",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=0.8, f_min=0, f_max=1.0),
                    ScalePart(p_min=0.8, p_max=1.0, f_min=8.0, f_max=9.0),
                ],
            ),
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 8.0, 9.0],
            id="two_different_parts",
        ),
        pytest.param(
            Scale(
                parts=[
                    ScalePart(p_min=0.0, p_max=0.8, f_min=100, f_max=110.0),
                    ScalePart(p_min=0.8, p_max=1.0, f_min=180.0, f_max=190.0),
                ],
            ),
            [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0],
            [100.0, 100.0, 100.0, 100.0, 110.0, 110.0, 110.0, 110.0, 180.0, 190.0],
            id="different_frequencies_same_indexes",
        ),
    ],
)
def test_frequency_scale_mapped_values(
    scale: Scale,
    original_scale: list[float],
    expected: list[int],
) -> None:
    assert scale.get_mapped_values(original_scale=original_scale) == expected


@pytest.mark.parametrize(
    ("input_matrix", "original_scale", "scale", "expected_matrix"),
    [
        pytest.param(
            np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]),
            [0.0, 1.0, 2.0, 3.0],
            Scale([ScalePart(0.0, 1.0, 0.0, 3.0)]),
            np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]),
            id="same_scale",
        ),
        pytest.param(
            np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]),
            [0.0, 1.0, 2.0, 3.0],
            Scale([ScalePart(0.0, 0.5, 0.0, 1.0), ScalePart(0.5, 1.0, 0.0, 1.0)]),
            np.array([[1, 10, 100], [2, 20, 200], [1, 10, 100], [2, 20, 200]]),
            id="repeat_first_half",
        ),
        pytest.param(
            np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]),
            [0.0, 1.0, 2.0, 3.0],
            Scale([ScalePart(0.0, 0.5, 2.0, 3.0), ScalePart(0.5, 1.0, 0.0, 1.0)]),
            np.array([[3, 30, 300], [4, 40, 400], [1, 10, 100], [2, 20, 200]]),
            id="switch_halves",
        ),
        pytest.param(
            np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300], [4, 40, 400]]),
            [0.0, 1.0, 2.0, 3.0],
            Scale(
                [
                    ScalePart(0.0, 0.25, 3.0, 4.0),
                    ScalePart(0.25, 0.5, 2.0, 3.0),
                    ScalePart(0.5, 0.75, 0.0, 1.0),
                    ScalePart(0.75, 1.0, 1.0, 2.0),
                ],
            ),
            np.array([[4, 40, 400], [3, 30, 300], [1, 10, 100], [2, 20, 200]]),
            id="four_parts",
        ),
    ],
)
def test_frequency_scale_rescale(
    input_matrix: np.ndarray,
    original_scale: np.ndarray,
    scale: Scale,
    expected_matrix: np.ndarray,
) -> None:
    scaled_matrix = scale.rescale(input_matrix, original_scale)
    assert np.array_equal(scaled_matrix, expected_matrix)


@pytest.mark.parametrize(
    ("part1", "part2", "expected"),
    [
        pytest.param(
            ScalePart(0.0, 1.0, 100, 500),
            ScalePart(0.0, 1.0, 100, 500),
            True,
            id="same_scale",
        ),
        pytest.param(
            ScalePart(0.0, 1.0, 100, 500, scale_type="log"),
            ScalePart(0.0, 1.0, 100, 500, scale_type="log"),
            True,
            id="same_log_scale",
        ),
        pytest.param(
            ScalePart(0.0, 1.0, 100, 500),
            ScalePart(0.0, 1.0, 100.0, 500.0),
            True,
            id="int_and_float_frequencies",
        ),
        pytest.param(
            ScalePart(0, 1, 100.0, 500.0),
            ScalePart(0.0, 1.0, 100.0, 500.0),
            True,
            id="int_and_float_p",
        ),
        pytest.param(
            ScalePart(0.05, 1.0, 100.0, 500.0),
            ScalePart(0.0, 1.0, 100.0, 500.0),
            False,
            id="different_p_min",
        ),
        pytest.param(
            ScalePart(0.0, 0.5, 100.0, 500.0),
            ScalePart(0.0, 1.0, 100.0, 500.0),
            False,
            id="different_p_max",
        ),
        pytest.param(
            ScalePart(0.0, 0.1, 150.0, 500.0),
            ScalePart(0.0, 1.0, 100.0, 500.0),
            False,
            id="different_f_min",
        ),
        pytest.param(
            ScalePart(0.0, 0.1, 100.0, 5000.0),
            ScalePart(0.0, 1.0, 100.0, 500.0),
            False,
            id="different_f_max",
        ),
        pytest.param(
            ScalePart(0.5, 0.6, 1000.0, 5000.0),
            ScalePart(0.0, 1.0, 100.0, 500.0, scale_type="log"),
            False,
            id="all_different",
        ),
        pytest.param(
            ScalePart(0.0, 1.0, 100, 500),
            ScalePart(0.0, 1.0, 100, 500, scale_type="log"),
            False,
            id="different_type",
        ),
    ],
)
def test_frequency_scale_part_equality(
    part1: ScalePart,
    part2: ScalePart,
    expected: bool,
) -> None:
    assert (part1 == part2) == expected
    assert (part2 == part1) == expected


@pytest.mark.parametrize(
    ("scale1", "scale2", "expected"),
    [
        pytest.param(
            Scale(
                [ScalePart(0.0, 1.0, 0.0, 1.0)],
            ),
            Scale(
                [ScalePart(0.0, 1.0, 0.0, 1.0)],
            ),
            True,
            id="same_scale_with_one_part",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                ],
            ),
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                ],
            ),
            True,
            id="same_scale_with_two_parts",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                ],
            ),
            Scale(
                [
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                ],
            ),
            True,
            id="order_doesnt_matter",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 10.0, 20.0),
                ],
            ),
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                ],
            ),
            False,
            id="one_different_part",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                ],
            ),
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0, scale_type="log"),
                    ScalePart(0.5, 1.0, 1.0, 2.0),
                ],
            ),
            False,
            id="one_different_type",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 1.0, 10.0, 20.0),
                ],
            ),
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 0.75, 1.0, 2.0),
                    ScalePart(0.75, 1.0, 2.0, 3.0),
                ],
            ),
            False,
            id="different_part_count",
        ),
    ],
)
def test_frequency_scale_equality(scale1: Scale, scale2: Scale, expected: bool) -> None:
    assert (scale1 == scale2) == expected
    assert (scale2 == scale1) == expected


@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 1.0, 0.0, 1.0),
                ],
            ),
            id="simple_scale",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                    ScalePart(0.5, 0.75, 1.0, 2.0),
                    ScalePart(0.75, 1.0, 2.0, 3.0),
                ],
            ),
            id="three_ordered_parts",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.75, 1.0, 2.0, 3.0),
                    ScalePart(0.5, 0.75, 1.0, 2.0),
                    ScalePart(0.0, 0.5, 0.0, 1.0),
                ],
            ),
            id="three_unordered_parts",
        ),
        pytest.param(
            Scale(
                [
                    ScalePart(0.0, 1.0, 0.0, 1.0, scale_type="log"),
                ],
            ),
            id="log_scale",
        ),
    ],
)
def test_frequency_scale_serialization(scale: Scale) -> None:
    assert Scale.from_dict_value(scale.to_dict_value()) == scale


@pytest.mark.parametrize(
    ("p_min", "p_max", "f_min", "f_max", "expected"),
    [
        pytest.param(
            -0.5,
            1.0,
            1.0,
            100.0,
            pytest.raises(
                ValueError,
                match="p_min must be between 0 and 1, got -0\\.5",
            ),
            id="negative_min",
        ),
        pytest.param(
            5.0,
            1.0,
            1.0,
            100.0,
            pytest.raises(
                ValueError,
                match="p_min must be between 0 and 1, got 5\\.0\n"
                "p_min must be strictly inferior than p_max, got \\(5\\.0,1\\.0\\)",
            ),
            id="min_too_big",
        ),
        pytest.param(
            0.0,
            -1.0,
            1.0,
            100.0,
            pytest.raises(
                ValueError,
                match="p_max must be between 0 and 1, got -1\\.0\n"
                "p_min must be strictly inferior than p_max, got \\(0\\.0,-1\\.0\\)",
            ),
            id="negative_max",
        ),
        pytest.param(
            0.0,
            2.0,
            1.0,
            100.0,
            pytest.raises(
                ValueError,
                match=r"p_max must be between 0 and 1, got 2.0",
            ),
            id="max_too_big",
        ),
        pytest.param(
            0.5,
            0.5,
            1.0,
            100.0,
            pytest.raises(
                ValueError,
                match="p_min must be strictly inferior than p_max,"
                " got \\(0\\.5,0\\.5\\)",
            ),
            id="p_min_equals_p_max",
        ),
        pytest.param(
            0.0,
            1.0,
            -1.0,
            100.0,
            pytest.raises(
                ValueError,
                match=r"f_min must be positive, got -1.0",
            ),
            id="negative_f_min",
        ),
        pytest.param(
            0.0,
            1.0,
            0.0,
            -100.0,
            pytest.raises(
                ValueError,
                match="f_max must be positive, got -100\\.0\n"
                "f_min must be strictly inferior than f_max, got \\(0\\.0,-100\\.0\\)",
            ),
            id="negative_f_max",
        ),
        pytest.param(
            0.0,
            1.0,
            500.0,
            500.0,
            pytest.raises(
                ValueError,
                match="f_min must be strictly inferior than f_max,"
                " got \\(500\\.0,500\\.0\\)",
            ),
            id="f_min_equals_f_max",
        ),
    ],
)
def test_scale_part_errors(
    p_min: float,
    p_max: float,
    f_min: float,
    f_max: float,
    expected: contextlib.AbstractContextManager,
) -> None:
    with expected as e:
        assert ScalePart(p_min=p_min, p_max=p_max, f_min=f_min, f_max=f_max) == e
