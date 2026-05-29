"""Tests for the conjecture-test helpers in ``minimal_region.py``.

The module is loaded directly from its file so the pure-Python tests run even
without Sage (importing the package ``hyperplane_arrangements`` would pull in
``sage.all`` via ``__init__``).  The single cone-bridge test is guarded with
``pytest.importorskip`` and runs only inside the Sage environment.
"""
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest

_MR_PATH = Path(__file__).resolve().parents[1] / "src" / "hyperplane_arrangements" / "minimal_region.py"
_spec = importlib.util.spec_from_file_location("hyperplane_arrangements.minimal_region", _MR_PATH)
mr = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mr
_spec.loader.exec_module(mr)


def _full(normals, m):
    """Minimal-chamber Solution at the full count ``m`` (matched as a multiset)."""
    results = mr.GreedyCutAllSolver(list(normals), list(m)).solve_all()
    target = sorted(m)
    return next(sol for counts, sol in results.items() if sorted(counts) == target)


def test_chamber_count_matches_solver():
    for normals, m in [
        ([(1, 0), (0, 1)], [3, 4]),
        ([(1, 0), (0, 1), (1, 1)], [2, 3, 2]),
        ([(1, 0), (0, 1), (1, 1), (1, -1)], [4, 4, 1, 1]),
    ]:
        sol = _full(normals, m)
        assert mr.chamber_count(sol.normals, sol.lines_by_dir) == sol.regions


def test_chamber_count_degenerate():
    assert mr.chamber_count([(1, 0)], [()]) == 1                       # no lines
    assert mr.chamber_count([(1, 0)], [(Fraction(0), Fraction(1))]) == 3  # 2 parallel lines


def test_minchamber_n2():
    for a1 in range(1, 5):
        for a2 in range(1, 5):
            sol = _full([(1, 0), (0, 1)], [a1, a2])
            assert sol.regions == mr.minchamber_n2((a1, a2)) == (a1 + 1) * (a2 + 1)


def test_minchamber_n3():
    for m in [(1, 1, 1), (2, 2, 1), (3, 2, 2), (2, 2, 2), (3, 3, 1)]:
        sol = _full([(1, 0), (0, 1), (1, 1)], m)
        assert sol.regions == mr.minchamber_n3(m)


def test_intersection_multiplicities_quasi_saturated():
    sol = _full([(1, 0), (0, 1), (1, 1), (1, -1)], [4, 4, 1, 1])  # p = 2
    assert mr.intersection_multiplicities(sol.normals, sol.lines_by_dir) == {2: 12, 3: 5, 4: 1}
    assert sol.regions == 36


def test_is_B2_type():
    B2 = [(1, 0), (0, 1), (1, 1), (1, -1)]
    assert mr.is_B2_type(B2)
    assert mr.is_B2_type([(2 * a + b, a + 3 * b) for a, b in B2])  # GL2 image (det 5)
    assert mr.is_B2_type([B2[i] for i in (2, 0, 3, 1)])           # permutation
    assert mr.is_B2_type([(-1, 0), (0, 1), (1, 1), (-1, 1)])      # sign variants
    assert not mr.is_B2_type([(1, 0), (0, 1), (1, 1), (1, 3)])    # generic
    assert not mr.is_B2_type([(1, 0), (0, 1), (1, 1)])            # only 3 directions


def test_is_quasi_saturated():
    qs = _full([(1, 0), (0, 1), (1, 1), (1, -1)], [4, 4, 1, 1])   # m=(2p,2p,1,1), p=2
    assert mr.is_quasi_saturated(qs.normals, qs.lines_by_dir)
    sat = _full([(1, 0), (0, 1), (1, 1), (1, -1)], [3, 3, 1, 1])  # saturated, odd counts
    assert not mr.is_quasi_saturated(sat.normals, sat.lines_by_dir)


def test_saturated_directions_P1_P2():
    # rem:m_not_enough: same m=(3,2,2,1), different direction sets.
    p1 = _full([(1, 0), (0, 1), (1, -1), (1, -2)], [3, 2, 2, 1])
    sat = mr.saturated_directions(p1.normals, p1.lines_by_dir)
    assert sat and p1.normals[sat[0]] == (1, 0)
    assert p1.regions == 24
    p2 = _full([(1, 0), (0, 1), (1, -1), (3, -1)], [3, 2, 2, 1])
    assert mr.saturated_directions(p2.normals, p2.lines_by_dir) == []
    assert p2.regions == 25


def test_s_invariant_matches_generator_size():
    for normals, m in [
        ([(1, 0), (0, 1), (1, 1)], [2, 2, 2]),
        ([(1, 0), (0, 1), (1, 1), (1, -1)], [4, 4, 1, 1]),
        ([(1, 0), (0, 1), (1, -1), (1, -2)], [3, 2, 2, 1]),
    ]:
        sol = _full(normals, m)
        assert mr.s_invariant(sol.normals, sol.lines_by_dir) == mr.generator_size(sol)
        assert mr.generator_size(sol) >= 3  # |P| >= 3  (rem:s-ge-3)


def test_closed_form_values():
    assert mr.saturation_lower_bound(8, 4) == Fraction(3)
    assert mr.saturation_chamber(3, 8) == 24
    assert mr.yoshinaga_bound(3, 5) == 24
    assert mr.minchamber_n2((3, 4)) == 20
    assert mr.minchamber_n3((2, 3, 2)) == 20


def test_cone_of_lines_freeness():
    pytest.importorskip("sage.all")
    from hyperplane_arrangements import HyperplaneArrangement

    # P1 is saturated with nonzero offsets, so the -c sign in the cone matters.
    sol = _full([(1, 0), (0, 1), (1, -1), (1, -2)], [3, 2, 2, 1])
    cone = HyperplaneArrangement.cone_of_lines(sol.normals, sol.lines_by_dir)
    assert bool(cone.is_free)
    assert sorted(int(d) for d in cone.degrees()) == [1, 3, 5]  # exp (1, a, N-a) = (1, 3, 5)
    assert mr.yoshinaga_bound(3, 5) == sol.regions == 24
