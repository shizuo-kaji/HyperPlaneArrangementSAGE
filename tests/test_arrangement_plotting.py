from sage.all import QQ, matrix

from hyperplane_arrangements import HyperplaneArrangement
from hyperplane_arrangements.arrangement_plotting import (
    affine_intersection,
    constructive_points,
    hyperplanes_through_flat,
    segment_from_row,
)


def test_segment_from_row_clips_affine_line():
    segment = segment_from_row([1, 0, -1], xlim=(0, 2), ylim=(0, 2))
    assert segment == ((1.0, 0), (1.0, 2))

    assert segment_from_row([0, 0, 1], xlim=(0, 2), ylim=(0, 2)) is None


def test_affine_intersection():
    point = affine_intersection([1, 0, -1], [0, 1, -2])
    assert point == (1.0, 2.0)

    assert affine_intersection([1, 0, -1], [2, 0, -3]) is None


def test_constructive_plotting_helpers_find_carriers():
    A = HyperplaneArrangement(
        matrix(
            QQ,
            [
                [1, 0, -1],
                [0, 1, -1],
                [1, -1, 0],
                [1, 1, -2],
                [1, 0, 0],
            ],
        )
    )

    assert hyperplanes_through_flat(A, (0, 1)) == (0, 1, 2, 3)
    points = constructive_points(A, (0, 1))
    assert len(points) == 1
    point, flat, carriers = points[0]
    assert point == (1.0, 1.0)
    assert flat == (0, 1)
    assert carriers == (0, 1, 2, 3)
