from hyperplane_arrangements import HyperplaneArrangement
from hyperplane_arrangements.saito_diagnostics import (
    height_bound_diagnostic,
    projective_dimension_D,
    scaled_minor_subset_diagnostics,
)


def test_projective_dimension_free_arrangement():
    A = HyperplaneArrangement([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert projective_dimension_D(A) == 0


def test_scaled_minor_subset_diagnostics_spog():
    A = HyperplaneArrangement([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 2]])
    report = scaled_minor_subset_diagnostics(A, verify=True)

    assert report["dimension"] == 3
    assert report["num_generators"] == 4
    assert report["num_subsets"] == 1
    assert report["num_criterion"] == 1
    assert report["num_actually_generates"] == 1


def test_height_bound_diagnostic_free_vacuous():
    A = HyperplaneArrangement([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    report = height_bound_diagnostic(A)

    assert report["projective_dimension"] == 0
    assert report["bound"] == 2
    assert report["ok"]
