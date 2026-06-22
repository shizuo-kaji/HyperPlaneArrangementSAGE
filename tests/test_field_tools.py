import numpy as np

from hyperplane_arrangements.field_tools import (
    boundary_samples,
    distance_to_segment,
    evaluate_polynomial_field,
    field_on_grid,
    fit_unconstrained_polynomial,
    grid_inside_polygon,
    min_distance_to_boundary,
    monomial_exponents,
    monomial_matrix,
    point_in_polygon,
    polygon_outline,
    sample_random_points_in_polygon,
    velocity_design_matrix,
)


def test_polygon_geometry_helpers():
    vertices = [(0, 0), (1, 0), (0, 1)]

    outline = polygon_outline(vertices)
    assert outline.shape == (4, 2)
    assert np.allclose(outline[0], outline[-1])

    assert point_in_polygon((0.2, 0.2), vertices)
    assert not point_in_polygon((1.2, 0.2), vertices)
    assert distance_to_segment((0.5, 1.0), (0, 0), (1, 0)) == 1.0
    assert min_distance_to_boundary((0.25, 0.25), vertices) > 0


def test_sampling_and_grid_helpers():
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

    pts = sample_random_points_in_polygon(vertices, 20, seed=123, min_dist_from_boundary=0.05)
    assert pts.shape == (20, 2)
    assert all(point_in_polygon(pt, vertices) for pt in pts)

    xs, ys, mask = grid_inside_polygon(vertices, nx=8, ny=9, pad_fraction=0)
    assert mask.shape == (9, 8)

    vx, vy, speed = field_on_grid(lambda p: np.column_stack((p[:, 0], -p[:, 1])), xs, ys, mask)
    assert vx.shape == mask.shape
    assert vy.shape == mask.shape
    assert speed.shape == mask.shape
    assert np.isnan(vx[~mask]).all()


def test_boundary_samples_shape_and_normals():
    vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
    points, normals, arclength = boundary_samples(vertices, samples_per_edge=3)

    assert points.shape == (12, 2)
    assert normals.shape == (12, 2)
    assert arclength.shape == (12,)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1)


def test_unconstrained_polynomial_fit_round_trips_linear_field():
    points = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [2, 1]], dtype=float)
    vectors = np.column_stack((1 + 2 * points[:, 0] - points[:, 1], 3 + points[:, 0] + 4 * points[:, 1]))

    assert monomial_exponents(1) == [(0, 0), (0, 1), (1, 0)]
    assert monomial_matrix(points, 1).shape == (5, 3)
    assert velocity_design_matrix(points, 1).shape == (10, 6)

    coeffs = fit_unconstrained_polynomial(points, vectors, degree=1)
    predicted = evaluate_polynomial_field(coeffs, points, degree=1)
    assert np.allclose(predicted, vectors)
