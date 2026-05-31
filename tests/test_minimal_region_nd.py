"""Tests for the general-dimension minimal-chamber solver ``minimal_region_nd.py``.

Pure-Python parts are loaded directly from file (like ``test_minimal_region.py``)
so they run without Sage; the Sage cross-checks are guarded with
``pytest.importorskip`` and run only inside the Sage environment.
"""
import importlib.util
import sys
from fractions import Fraction
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src" / "hyperplane_arrangements"


def _load(modname, filename):
    spec = importlib.util.spec_from_file_location(modname, _SRC / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


mrnd = _load("hyperplane_arrangements.minimal_region_nd", "minimal_region_nd.py")
mr = _load("hyperplane_arrangements.minimal_region", "minimal_region.py")

F = Fraction


# --------------------------------------------------------------------------- #
# region_count_nd
# --------------------------------------------------------------------------- #

def test_region_count_known_values():
    rc = mrnd.region_count_nd
    assert rc([], []) == 1
    # boolean R^3
    assert rc([(1, 0, 0), (0, 1, 0), (0, 0, 1)], [[F(0)], [F(0)], [F(0)]]) == 8
    # three planes with independent normals always meet at one point -> 8
    assert rc([(1, 0, 0), (0, 1, 0), (1, 1, 1)], [[F(0)], [F(0)], [F(0)]]) == 8
    # two parallel x-planes (3 slabs) times one y-cut (2) = 6
    assert rc([(1, 0, 0), (0, 1, 0)], [[F(0), F(1)], [F(0)]]) == 6
    # boolean R^4 = 16
    assert rc([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)],
              [[F(0)]] * 4) == 16


def test_region_count_generic_position_r3():
    # N planes in general position in R^ell: sum_{k<=ell} C(N, k)
    from math import comb
    normals = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3), (3, 1, 2)]
    offsets = [(F(1),), (F(2),), (F(3),), (F(5),), (F(7),)]  # generic offsets
    N = len(normals)
    expected = sum(comb(N, k) for k in range(0, 4))  # 1 + 5 + 10 + 10 = 26
    assert mrnd.region_count_nd(normals, offsets) == expected


def test_region_count_matches_2d_chamber_count():
    import random
    random.seed(0)
    normals = [(1, 0), (0, 1), (1, 1), (1, -1)]
    for _ in range(40):
        offs = tuple(
            tuple(sorted(set(F(random.randint(-2, 2)) for _ in range(random.randint(0, 3)))))
            for _ in normals
        )
        assert mrnd.region_count_nd(normals, offs) == mr.chamber_count(tuple(normals), offs)


# --------------------------------------------------------------------------- #
# Solver: ell=2 regression
# --------------------------------------------------------------------------- #

def test_minchamber_nd_matches_closed_forms():
    assert mrnd.minchamber_nd([(1, 0), (0, 1)], [3, 4]) == mr.minchamber_n2((3, 4))
    for m in ([2, 3, 2], [3, 3, 3], [4, 2, 1]):
        assert mrnd.minchamber_nd([(1, 0), (0, 1), (1, 1)], m) == mr.minchamber_n3(m)


def test_minchamber_nd_matches_2d_solver():
    def full2(normals, m):
        res = mr.GreedyCutAllSolver(list(normals), list(m)).solve_all()
        t = sorted(m)
        return next(s for c, s in res.items() if sorted(c) == t).regions

    for normals, m in [
        ([(1, 0), (0, 1)], [3, 4]),
        ([(1, 0), (0, 1), (1, 1)], [2, 2, 2]),
        ([(1, 0), (0, 1), (1, 1), (1, -1)], [2, 2, 1, 1]),
    ]:
        assert mrnd.minchamber_nd(normals, m) == full2(normals, m)


def test_full_rank_required():
    # directions spanning only a 2-plane inside R^3 -> degenerate, must raise
    with pytest.raises(ValueError):
        mrnd.GreedyCutAllSolverND([(1, 0, 0), (0, 1, 0), (1, 1, 0)], [1, 1, 1])


# --------------------------------------------------------------------------- #
# Solver vs exhaustive oracle (small ell=3 instances)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("normals,m", [
    ([(1, 0, 0), (0, 1, 0), (0, 0, 1)], [1, 1, 1]),
    ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], [1, 1, 1, 1]),
    ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)], [1, 1, 1, 1]),
    ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1), (1, -1, 0)], [1, 1, 1, 1, 1]),
])
def test_greedy_matches_exhaustive_ell3(normals, m):
    mrnd.assert_greedy_optimal(normals, m)


# --------------------------------------------------------------------------- #
# Sage cross-checks
# --------------------------------------------------------------------------- #

def test_region_count_matches_sage():
    pytest.importorskip("sage.all")
    from hyperplane_arrangements import HyperplaneArrangement as HA
    cases = [
        ([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)], [[F(0)], [F(0)], [F(0)], [F(-1)]]),
        ([(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1)], [[F(0), F(2)], [F(0)], [F(1)], [F(0), F(3)]]),
        ([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (1, 1, 1, 1)],
         [[F(0)], [F(1)], [F(2)], [F(3)], [F(0)]]),
    ]
    for normals, offsets in cases:
        assert mrnd.region_count_nd(normals, offsets) == \
            HA.n_regions_of_arrangement(normals, offsets)


def test_cone_roundtrip_and_chamber():
    pytest.importorskip("sage.all")
    from hyperplane_arrangements import HyperplaneArrangement as HA

    normals = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    m = [2, 1, 1, 1]
    sol = mrnd.solve_minimal_nd(normals, m)
    A_min = HA.cone_of_arrangement(sol.normals, sol.offsets_by_dir)

    # cone round-trips through restriction to H_0 = {x_0 = 0}
    P = A_min.restriction([1, 0, 0, 0])
    recov = sorted(mrnd.normalize_normal_nd(tuple(int(x) for x in r)) for r in P.mat.rows())
    orig = sorted(mrnd.normalize_normal_nd(n) for n in normals)
    assert recov == orig
    assert sum(int(x) for x in P.multiplicity) == sum(m)

    # chamber(c(B)) = 2 r(B) = Sage n_regions(A_min)
    r = mrnd.region_count_nd(sol.normals, sol.offsets_by_dir)
    assert r == HA.n_regions_of_arrangement(sol.normals, sol.offsets_by_dir)
    assert int(A_min.n_regions()) == 2 * r


def test_minimal_arrangement_end_to_end():
    pytest.importorskip("sage.all")
    from hyperplane_arrangements import HyperplaneArrangement as HA

    normals = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1)]
    m = [2, 1, 1, 1]
    Bmin = mrnd.solve_minimal_nd(normals, m)
    A_min = HA.cone_of_arrangement(Bmin.normals, Bmin.offsets_by_dir)

    # A_min is minimal
    res = A_min.minimal_arrangement([1, 0, 0, 0])
    assert res.is_A_minimal is True
    assert res.chamber == 2 * Bmin.regions

    # a generic cone with the same (P, m) is NOT minimal (more chambers)
    A_gen = HA.cone_of_arrangement(
        normals, [[F(0), F(7)], [F(0)], [F(0)], [F(13)]]
    )
    res2 = A_gen.minimal_arrangement([1, 0, 0, 0])
    assert res2.is_A_minimal is False
    assert int(A_gen.n_regions()) > res2.chamber
