r"""Minimal-chamber affine hyperplane arrangements in general dimension ``ell``.

This is the general-dimension counterpart of :mod:`minimal_region` (which is
specialised to 2D line arrangements).  It supports the experiment of
``doc/minimumregion_conjecture.tex``:

    Given a central arrangement ``A`` in ``R^{ell+1}`` and a restriction plane
    ``H`` (the doc fixes ``H = {x_0 = 0}``), the restriction ``A^H`` is an
    ``ell``-dimensional central multi-arrangement ``(P, m)``.  ``Arr(P, m)`` is
    the family of affine arrangements in ``R^ell`` realising the directions
    ``P`` with the multiplicities ``m`` and arbitrary parallel offsets.  An
    arrangement ``B in Arr(P, m)`` is *minimal* when its cone ``c(B)`` has the
    fewest chambers.

Since ``chamber(c(B)) = 2 * r(B)`` where ``r(B)`` is the number of regions of
the affine arrangement ``B`` in ``R^ell``, minimising chambers is the same as
minimising ``r(B)``.  A hyperplane of a fixed direction ``alpha in P`` is
exactly ``{x : alpha . x = c}`` for a single scalar offset ``c``, so the search
space has the same shape as the 2D solver -- only the flats and the region
count are now genuinely ``ell``-dimensional.

Design (all exact, ``Fraction``/``int`` only -- never float):

* :func:`region_count_nd` -- the authoritative region counter via the
  intersection semilattice and its Moebius function (Zaslavsky).  Cross-checked
  against the 2D formula and against Sage's ``n_regions`` in the tests.
* :class:`GreedyCutAllSolverND` -- a legal-move depth-first search generalising
  :class:`minimal_region.GreedyCutAllSolver`.  It explores every arrangement
  reachable by placing each new hyperplane through an existing flat (or a
  generic seed), exactly as the 2D solver does.  **The optimality of this
  legal-move search is proven only for ``ell = 2``** (see ``conjectures.tex``);
  for ``ell >= 3`` it is a heuristic upper bound on ``minchamber(P, m)``, to be
  confirmed by :func:`min_region_exhaustive` on small instances.
* :func:`min_region_exhaustive` -- an independent brute-force oracle over a
  bounded rational offset pool, used to validate the greedy search.

The module is pure-Python (stdlib only) so it can be imported without Sage; the
Sage bridge (cone construction, ``n_regions`` cross-check, the high-level
``minimal_arrangement(A, H)`` pipeline) lives in :mod:`arrangement`.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

NormalND = Tuple[int, ...]
PointND = Tuple[Fraction, ...]
Offset = Fraction

ZERO = Fraction(0)
ONE = Fraction(1)


# --------------------------------------------------------------------------- #
# Direction normalisation and merging
# --------------------------------------------------------------------------- #

def normalize_normal_nd(vec: Sequence[int]) -> NormalND:
    """Canonicalise an integer direction: divide by gcd, first nonzero > 0."""
    entries = [int(x) for x in vec]
    if all(x == 0 for x in entries):
        raise ValueError("the zero vector is not a valid normal")
    g = 0
    for x in entries:
        g = gcd(g, abs(x))
    entries = [x // g for x in entries]
    for x in entries:
        if x != 0:
            if x < 0:
                entries = [-y for y in entries]
            break
    return tuple(entries)


def merge_normals_nd(
    normals: Iterable[Sequence[int]],
    max_counts: Iterable[int],
) -> Tuple[List[NormalND], List[int]]:
    """Normalise directions and merge duplicates, summing their multiplicities."""
    normals = list(normals)
    max_counts = list(max_counts)
    if len(normals) != len(max_counts):
        raise ValueError("normals and max_counts must have the same length")
    if not normals:
        return [], []
    ell = len(tuple(normals[0]))
    agg: Dict[NormalND, int] = {}
    order: List[NormalND] = []
    for vec, mc in zip(normals, max_counts):
        if len(tuple(vec)) != ell:
            raise ValueError("all normals must have the same dimension")
        nn = normalize_normal_nd(vec)
        if nn not in agg:
            agg[nn] = 0
            order.append(nn)
        agg[nn] += int(mc)
    return order, [agg[n] for n in order]


def _to_fraction(x) -> Fraction:
    """Coerce to a *clean* python :class:`Fraction`.

    Accepts ``int``, clean ``Fraction``, Sage ``Integer``/``Rational``, and even a
    *broken* ``Fraction`` built as ``Fraction(sage_value)`` -- Sage numbers expose
    ``numerator``/``denominator`` as *methods*, so ``Fraction(sage_int)`` stores a
    bound method as its numerator.  We detect that and rebuild, so the rest of the
    module stays pure-Python while tolerating Sage inputs from the bridge.
    """
    if type(x) is Fraction and isinstance(x.numerator, int):
        return x
    num = getattr(x, "numerator", None)
    den = getattr(x, "denominator", None)
    if num is not None and den is not None:
        if callable(num):
            num = num()
        if callable(den):
            den = den()
        return Fraction(int(num), int(den))
    if isinstance(x, int):
        return Fraction(x)
    return Fraction(x)


def _dot(a: Sequence, b: Sequence) -> Fraction:
    r"""
    Internal helper method `_dot`.
    """
    return sum((_to_fraction(x) * _to_fraction(y) for x, y in zip(a, b)), ZERO)


# --------------------------------------------------------------------------- #
# Exact rational linear algebra: flats as RREF of an augmented system
# --------------------------------------------------------------------------- #

def _rref_augmented(
    rows: Sequence[Sequence[Fraction]], ell: int
) -> Tuple[Tuple[Tuple[Fraction, ...], ...], List[int], bool]:
    """Reduced row echelon form of an augmented matrix ``[N | b]`` (``ell+1`` cols).

    Returns ``(nonzero_rows, var_pivot_cols, consistent)`` where ``nonzero_rows``
    are the canonical reduced rows (each of length ``ell + 1``), ``var_pivot_cols``
    are the pivot columns among the first ``ell`` (variable) columns, and
    ``consistent`` is ``False`` iff a pivot lands in the right-hand-side column
    (an empty solution set).
    """
    ncols = ell + 1
    M = [[_to_fraction(x) for x in row] for row in rows]
    pivots: List[int] = []
    pr = 0
    for c in range(ncols):
        piv = None
        for i in range(pr, len(M)):
            if M[i][c] != 0:
                piv = i
                break
        if piv is None:
            continue
        M[pr], M[piv] = M[piv], M[pr]
        pivval = M[pr][c]
        M[pr] = [x / pivval for x in M[pr]]
        for i in range(len(M)):
            if i != pr and M[i][c] != 0:
                f = M[i][c]
                M[i] = [a - f * b for a, b in zip(M[i], M[pr])]
        pivots.append(c)
        pr += 1
        if pr == len(M):
            break
    nonzero = tuple(tuple(M[i]) for i in range(pr))
    consistent = ell not in pivots
    var_pivots = [c for c in pivots if c < ell]
    return nonzero, var_pivots, consistent


@dataclass(frozen=True)
class Flat:
    """A nonempty affine flat = solution set of ``N x = b``, in canonical form.

    ``key`` is the tuple of RREF rows (each ``length ell + 1``) and is the exact
    de-duplication key: two systems define the same flat iff their keys match.
    ``rank`` is the codimension; ``witness`` is a particular point;
    ``hull`` is a basis of the flat's direction space (so ``dim = ell - rank``).
    """
    key: Tuple[Tuple[Fraction, ...], ...]
    rank: int
    witness: PointND
    hull: Tuple[PointND, ...]


def _make_flat(rows: Sequence[Sequence[Fraction]], ell: int) -> Optional[Flat]:
    """Build the canonical :class:`Flat` from augmented rows, or ``None`` if empty."""
    nonzero, var_pivots, consistent = _rref_augmented(rows, ell)
    if not consistent:
        return None
    rank = len(var_pivots)
    witness = [ZERO] * ell
    for idx, c in enumerate(var_pivots):
        witness[c] = nonzero[idx][ell]
    free_cols = [c for c in range(ell) if c not in var_pivots]
    hull: List[PointND] = []
    for f in free_cols:
        d = [ZERO] * ell
        d[f] = ONE
        for idx, c in enumerate(var_pivots):
            d[c] = -nonzero[idx][f]
        hull.append(tuple(d))
    return Flat(key=nonzero, rank=rank, witness=tuple(witness), hull=tuple(hull))


def _whole_space(ell: int) -> Flat:
    r"""
    Internal helper method `_whole_space`.
    """
    hull = tuple(
        tuple(ONE if j == i else ZERO for j in range(ell)) for i in range(ell)
    )
    return Flat(key=(), rank=0, witness=tuple([ZERO] * ell), hull=hull)


def _hyperplane_rows(
    normals: Sequence[NormalND],
    offsets_by_dir: Sequence[Iterable[Fraction]],
) -> List[Tuple[Fraction, ...]]:
    """Flatten ``(normals, offsets_by_dir)`` into augmented rows ``[alpha | c]``."""
    rows: List[Tuple[Fraction, ...]] = []
    for alpha, offsets in zip(normals, offsets_by_dir):
        alpha_frac = tuple(_to_fraction(a) for a in alpha)
        for c in offsets:
            rows.append(alpha_frac + (_to_fraction(c),))
    return rows


def build_flats(
    normals: Sequence[NormalND],
    offsets_by_dir: Sequence[Iterable[Fraction]],
    ell: Optional[int] = None,
) -> List[Flat]:
    """Return all nonempty flats of the affine arrangement (the intersection poset).

    Includes the whole space ``R^ell`` (the bottom ``0hat``) and excludes empty
    intersections.  Built by closing the whole space under intersection with
    every hyperplane (so every subset-intersection appears).
    """
    if ell is None:
        ell = len(normals[0]) if normals else 0
    rows = _hyperplane_rows(normals, offsets_by_dir)

    flats: Dict[Tuple, Flat] = {}
    bottom = _whole_space(ell)
    flats[bottom.key] = bottom
    worklist: List[Flat] = [bottom]
    while worklist:
        F = worklist.pop()
        for h in rows:
            new_rows = [list(r) for r in F.key] + [list(h)]
            G = _make_flat(new_rows, ell)
            if G is None:
                continue
            if G.key not in flats:
                flats[G.key] = G
                worklist.append(G)
    return list(flats.values())


def _flat_contains_flat(F: Flat, G: Flat, ell: int) -> bool:
    """``True`` iff ``sol(G) subset sol(F)`` -- every equation of ``F`` holds on ``G``."""
    for row in F.key:
        normal = row[:ell]
        rhs = row[ell]
        if _dot(normal, G.witness) != rhs:
            return False
        for d in G.hull:
            if _dot(normal, d) != 0:
                return False
    return True


def mobius_region_count(flats: Sequence[Flat], ell: int) -> int:
    r"""Number of regions via Zaslavsky: ``r = sum_F |mu(0hat, F)|``.

    ``mu`` is the Moebius function of the intersection poset ordered by reverse
    inclusion of solution sets (bottom ``0hat = R^ell``).  Holds for affine
    arrangements; counts bounded and unbounded regions together.
    """
    order = sorted(range(len(flats)), key=lambda i: flats[i].rank)
    mu: List[int] = [0] * len(flats)
    by_rank: List[List[int]] = []
    for idx in order:
        F = flats[idx]
        if F.rank == 0:
            mu[idx] = 1
        else:
            s = 0
            for jdx in order:
                G = flats[jdx]
                if G.rank >= F.rank:
                    break
                if _flat_contains_flat(G, F, ell):  # G < F strictly (lower rank)
                    s += mu[jdx]
            mu[idx] = -s
    return sum(abs(v) for v in mu)


def region_count_nd(
    normals: Sequence[NormalND],
    offsets_by_dir: Sequence[Iterable[Fraction]],
) -> int:
    """Exact number of regions of the affine arrangement in ``R^ell``."""
    if not normals:
        return 1
    ell = len(normals[0])
    flats = build_flats(normals, offsets_by_dir, ell)
    return mobius_region_count(flats, ell)


def _rank_normals(normals: Sequence[NormalND], ell: int) -> int:
    r"""
    Internal helper method `_rank_normals`.
    """
    rows = [tuple(_to_fraction(a) for a in alpha) + (ZERO,) for alpha in normals]
    _, var_pivots, _ = _rref_augmented(rows, ell)
    return len(var_pivots)


# --------------------------------------------------------------------------- #
# Minimal-configuration search
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SolutionND:
    r"""A minimal-chamber affine arrangement for a fixed direction multiset.

    Attributes:
        regions: the region count ``r(B)`` of this affine arrangement (the cone
            ``c(B)`` then has ``chamber = 2 * regions``).
        normals: the distinct (merged, canonicalised) direction normals.
        offsets_by_dir: offsets of the hyperplanes grouped by direction.
        seed_points: generic seed vertices introduced by the search (diagnostic).
    """
    regions: int
    normals: Tuple[NormalND, ...]
    offsets_by_dir: Tuple[Tuple[Fraction, ...], ...]
    seed_points: Tuple[PointND, ...] = ()


class GreedyCutAllSolverND:
    r"""Legal-move depth-first search for the minimal region count in ``R^ell``.

    Generalises :class:`minimal_region.GreedyCutAllSolver` to general dimension.
    Starting from the empty arrangement it repeatedly adds a hyperplane of an
    available direction, placing it through an existing flat (a "legal move") or,
    when none remain, through a freshly introduced generic seed vertex.  Every
    reachable configuration is explored; :attr:`best_for_counts` keeps the
    minimum region count seen for each count vector.

    Optimality note: for ``ell = 2`` the legal-move search is known to reach a
    global minimiser (``conjectures.tex``).  **For ``ell >= 3`` this is a
    heuristic** -- it returns an upper bound on ``minchamber(P, m)``; confirm it
    against :func:`min_region_exhaustive` on small instances.

    Unlike the 2D solver (which hard-requires the direction ``(0, 1)``), this
    solver only requires the directions to span ``R^ell`` (so 0-dimensional
    flats, i.e. vertices, can occur).
    """

    def __init__(
        self,
        normals: Iterable[Sequence[int]],
        max_counts: Iterable[int],
        *,
        max_seed_radius: int = 6,
    ) -> None:
        r"""
        Initialize the object.
        """
        normals, max_counts = merge_normals_nd(normals, max_counts)
        if not normals:
            raise ValueError("at least one direction is required")
        self.normals: Tuple[NormalND, ...] = tuple(normals)
        self.max_counts: Tuple[int, ...] = tuple(max_counts)
        self.m = len(self.normals)
        self.ell = len(self.normals[0])
        if _rank_normals(self.normals, self.ell) != self.ell:
            raise ValueError(
                "directions must span R^ell (need a full-rank set of normals); "
                "the problem otherwise degenerates to a lower dimension"
            )
        self.max_seed_radius = int(max_seed_radius)
        self.origin: PointND = tuple([ZERO] * self.ell)
        self.best_for_counts: Dict[
            Tuple[int, ...],
            Tuple[int, Tuple[Tuple[Fraction, ...], ...], Tuple[PointND, ...]],
        ] = {}
        self.visited: Set[Tuple] = set()

    def solve_all(self) -> Dict[Tuple[int, ...], SolutionND]:
        r"""
        Solve and explore all legal configurations.

        OUTPUT:

        - A dictionary or result structure containing minimal configurations.
        """
        empty = tuple(frozenset() for _ in range(self.m))
        self._dfs(empty, frozenset({self.origin}))
        result: Dict[Tuple[int, ...], SolutionND] = {}
        for counts, (regions, offsets, seeds) in self.best_for_counts.items():
            if all(c == 0 for c in counts):
                continue
            offs_sorted = tuple(tuple(sorted(s)) for s in offsets)
            seed_pts = tuple(sorted(p for p in seeds if p != self.origin))
            result[counts] = SolutionND(int(regions), self.normals, offs_sorted, seed_pts)
        return result

    def _record(self, counts, regions, offsets, seeds) -> None:
        r"""
        Internal helper method `_record`.
        """
        current = self.best_for_counts.get(counts)
        if current is None or regions < current[0]:
            self.best_for_counts[counts] = (regions, offsets, seeds)

    def _dfs(
        self,
        offsets: Tuple[frozenset, ...],
        seeds: frozenset,
    ) -> None:
        r"""
        Internal helper method `_dfs`.
        """
        state_key = (offsets, seeds)
        if state_key in self.visited:
            return
        self.visited.add(state_key)

        flats = build_flats(self.normals, offsets, self.ell)
        regions = mobius_region_count(flats, self.ell)
        counts = tuple(len(s) for s in offsets)
        self._record(counts, regions, offsets, seeds)

        active = [i for i in range(self.m) if counts[i] < self.max_counts[i]]
        if not active:
            return

        # Candidate offsets: a hyperplane of direction alpha_i is a legal move if
        # it contains an existing flat F (alpha_i constant on F's hull), at offset
        # c = alpha_i . witness(F).  Seed vertices contribute likewise.  The
        # "score" counts the coincidences and is used only for greedy ordering.
        moves: List[Tuple[int, int, Fraction]] = []
        for i in active:
            alpha = self.normals[i]
            cand: Dict[Fraction, int] = {}
            for F in flats:
                if all(_dot(alpha, d) == 0 for d in F.hull):
                    c = _dot(alpha, F.witness)
                    cand[c] = cand.get(c, 0) + 1
            for s in seeds:
                c = _dot(alpha, s)
                cand[c] = cand.get(c, 0) + 1
            for c, score in cand.items():
                if c in offsets[i]:
                    continue
                moves.append((score, i, c))

        if not moves:
            seed = self._next_seed_vertex(offsets, flats, seeds, active)
            if seed is None or seed in seeds:
                return
            self._dfs(offsets, seeds | frozenset({seed}))
            return

        moves.sort(key=lambda t: -t[0])  # greedy: most coincident placements first
        for _score, i, c in moves:
            new_offsets = list(offsets)
            new_offsets[i] = offsets[i] | frozenset({c})
            self._dfs(tuple(new_offsets), seeds)

    def _next_seed_vertex(self, offsets, flats, seeds, active) -> Optional[PointND]:
        """A generic lattice point off every hyperplane and existing vertex offset."""
        used_offsets = [set(o) for o in offsets]
        # offsets that would merely re-hit an existing vertex (avoid duplicating moves)
        vertices = [F.witness for F in flats if F.rank == self.ell]
        blocked = {
            i: {_dot(self.normals[i], v) for v in vertices} for i in active
        }
        for radius in range(self.max_seed_radius + 1):
            for pt in self._lattice_shell(radius):
                if pt in seeds:
                    continue
                if any(_dot(self.normals[j], pt) in used_offsets[j] for j in range(self.m)):
                    continue
                if all(_dot(self.normals[i], pt) not in blocked[i] for i in active):
                    return pt
        return None

    def _lattice_shell(self, radius: int) -> Iterable[PointND]:
        r"""
        Internal helper method `_lattice_shell`.
        """
        if radius == 0:
            yield tuple([ZERO] * self.ell)
            return
        import itertools as _it
        rng = range(-radius, radius + 1)
        for coords in _it.product(rng, repeat=self.ell):
            if max(abs(c) for c in coords) == radius:
                yield tuple(Fraction(c) for c in coords)


def minchamber_nd(
    normals: Iterable[Sequence[int]],
    max_counts: Iterable[int],
    **kwargs,
) -> int:
    r"""Return ``minchamber(P, m)`` -- the minimal region count over ``Arr(P, m)``
    as found by :class:`GreedyCutAllSolverND` (an upper bound for ``ell >= 3``)."""
    solver = GreedyCutAllSolverND(normals, max_counts, **kwargs)
    results = solver.solve_all()
    target = solver.max_counts
    sol = results.get(target)
    if sol is None:
        for counts, s in results.items():
            if sorted(counts) == sorted(target):
                sol = s
                break
    if sol is None:
        raise RuntimeError("solver did not reach the full count vector")
    return sol.regions


def solve_minimal_nd(
    normals: Iterable[Sequence[int]],
    max_counts: Iterable[int],
    **kwargs,
) -> SolutionND:
    """Like :func:`minchamber_nd` but return the full :class:`SolutionND`."""
    solver = GreedyCutAllSolverND(normals, max_counts, **kwargs)
    results = solver.solve_all()
    target = solver.max_counts
    sol = results.get(target)
    if sol is None:
        for counts, s in results.items():
            if sorted(counts) == sorted(target):
                sol = s
                break
    if sol is None:
        raise RuntimeError("solver did not reach the full count vector")
    return sol


# --------------------------------------------------------------------------- #
# Exhaustive verifier (small instances) -- an independent oracle for the greedy
# --------------------------------------------------------------------------- #

def min_region_exhaustive(
    normals: Iterable[Sequence[int]],
    max_counts: Iterable[int],
    *,
    pool: Optional[Sequence[int]] = None,
    max_configs: int = 200_000,
) -> Tuple[int, SolutionND]:
    r"""Brute-force minimum region count over a bounded integer offset pool.

    For each direction with multiplicity ``a_i`` it chooses ``a_i`` distinct
    offsets from ``pool`` (default ``0 .. N``) and counts regions of every
    resulting arrangement, keeping the minimum.  Region count is
    translation-invariant, so (i) we may assume ``0`` is among the first
    direction's offsets (a translation gauge, which cuts the search) and (ii) a
    bounded integer pool realises every incidence pattern provided it is rich
    enough -- which it is for the small instances where this is feasible.  Raises
    if the number of configurations exceeds ``max_configs``.

    Used as an independent check on :class:`GreedyCutAllSolverND` (see
    :func:`assert_greedy_optimal`).
    """
    import itertools as _it
    from math import comb

    normals, max_counts = merge_normals_nd(normals, max_counts)
    N = sum(max_counts)
    if pool is None:
        pool = list(range(N + 1))
    pool_f = [Fraction(int(p)) for p in pool]
    zero = Fraction(0)

    choices: List[List[Tuple[Fraction, ...]]] = []
    total = 1
    for idx, a in enumerate(max_counts):
        if a > len(pool_f):
            raise ValueError(f"pool too small for multiplicity {a}")
        combos = list(_it.combinations(pool_f, a))
        if idx == 0 and zero in pool_f:
            # translation gauge: WLOG some hyperplane of the first direction sits at 0
            combos = [c for c in combos if zero in c]
        choices.append(combos)
        total *= len(combos)
    if total > max_configs:
        raise ValueError(
            f"{total} configurations exceed max_configs={max_configs}; "
            "increase max_configs or shrink the instance/pool"
        )
    best: Optional[int] = None
    best_offsets: Optional[Tuple[Tuple[Fraction, ...], ...]] = None
    for combo in _it.product(*choices):
        offsets = tuple(tuple(c) for c in combo)
        r = region_count_nd(normals, offsets)
        if best is None or r < best:
            best = r
            best_offsets = offsets
    assert best is not None and best_offsets is not None
    return best, SolutionND(best, tuple(normals), best_offsets, ())


def assert_greedy_optimal(
    normals: Iterable[Sequence[int]],
    max_counts: Iterable[int],
    *,
    pool: Optional[Sequence[int]] = None,
    max_configs: int = 200_000,
) -> int:
    """Assert the greedy search matches the exhaustive oracle; return the value."""
    greedy = minchamber_nd(normals, max_counts)
    exact, _ = min_region_exhaustive(
        normals, max_counts, pool=pool, max_configs=max_configs
    )
    if greedy != exact:
        verdict = "greedy MISSED the minimum" if greedy > exact else "pool too small"
        raise AssertionError(
            f"greedy={greedy} != exhaustive={exact} for normals={list(normals)}, "
            f"max_counts={list(max_counts)} ({verdict})"
        )
    return greedy


# --------------------------------------------------------------------------- #
# Affine constructive closure / s_k (no cone, no infinity)
# --------------------------------------------------------------------------- #
#
# This is the *affine* L_k constructive closure of an arrangement B in R^ell
# (the convention of ``doc/conjectures.tex`` and the 2D ``minimal_region.py``),
# as opposed to the *central* L_k closure of the cone c(B) computed by
# ``arrangement.HyperplaneArrangement.s_invariant`` (which also sees H_0 and the
# flats at infinity).  The two can differ; ``doc/minimumregion_conjecture.tex``'s
# s(A) is taken here in the affine sense (s_k with k = 2).

def affine_hyperplanes(
    normals: Sequence[NormalND],
    offsets_by_dir: Sequence[Iterable[Fraction]],
) -> List[Tuple[int, Fraction]]:
    """Flatten ``(normals, offsets_by_dir)`` into a list of ``(dir_index, offset)``."""
    return [(i, _to_fraction(c)) for i, offs in enumerate(offsets_by_dir) for c in offs]


def _entry_contains_flat(normals, entries, t, F: Flat) -> bool:
    r"""
    Internal helper method `_entry_contains_flat`.
    """
    i, c = entries[t]
    alpha = normals[i]
    if _dot(alpha, F.witness) != c:
        return False
    return all(_dot(alpha, d) == 0 for d in F.hull)


def constructive_closure_affine(
    normals: Sequence[NormalND],
    offsets_by_dir: Sequence[Iterable[Fraction]],
    seed: Iterable[int],
    k: int = 2,
) -> frozenset:
    r"""``L_k`` constructive closure of ``seed`` for the affine arrangement.

    ``seed`` and the result are sets of indices into
    :func:`affine_hyperplanes`.  The closure adds, for every rank-``k`` flat
    ``X`` of the current set and every hyperplane ``H`` of the arrangement with
    ``X \subset H``, the hyperplane ``H`` -- repeated to a fixed point.
    """
    ell = len(normals[0])
    entries = affine_hyperplanes(normals, offsets_by_dir)
    p = len(entries)
    current = set(seed)
    if k <= 0:
        return frozenset(current)
    changed = True
    while changed:
        changed = False
        cur = sorted(current)
        if len(cur) < k:
            break
        for combo in itertools.combinations(cur, k):
            rows = [[*(_to_fraction(a) for a in normals[entries[t][0]]), entries[t][1]]
                    for t in combo]
            F = _make_flat(rows, ell)
            if F is None or F.rank != k:
                continue
            for t in range(p):
                if t not in current and _entry_contains_flat(normals, entries, t, F):
                    current.add(t)
                    changed = True
    return frozenset(current)


def s_invariant_affine(
    normals: Sequence[NormalND],
    offsets_by_dir: Sequence[Iterable[Fraction]],
    k: int = 2,
) -> int:
    r"""Affine ``s_k``: the minimum number of hyperplanes whose ``L_k`` affine
    constructive closure is the whole arrangement.

    This is the affine analogue of
    :meth:`arrangement.HyperplaneArrangement.s_invariant`; for the
    ``minimumregion_conjecture.tex`` experiment use ``k = 2``.
    """
    entries = affine_hyperplanes(normals, offsets_by_dir)
    p = len(entries)
    if p == 0:
        return 0
    full = frozenset(range(p))
    # a seed of size < k carries no rank-k flat, so it cannot grow
    min_size = k if (k > 1 and p >= k) else p
    if min_size < 1:
        min_size = 1
    for size in range(min_size, p + 1):
        for seed in itertools.combinations(range(p), size):
            if constructive_closure_affine(normals, offsets_by_dir, seed, k) == full:
                return size
    return p


# --------------------------------------------------------------------------- #
# Enumerate ALL minimal configurations (for the for-all-minimal conjecture)
# --------------------------------------------------------------------------- #

def enumerate_minimal_configs(
    normals: Iterable[Sequence[int]],
    max_counts: Iterable[int],
    *,
    pool: Optional[Sequence[int]] = None,
    max_configs: int = 200_000,
) -> Tuple[int, List[NormalND], List[Tuple[Tuple[Fraction, ...], ...]]]:
    r"""Return ``(min_regions, normals, configs)`` where ``configs`` are **all**
    offset assignments over the pool that achieve the minimal region count.

    Same enumeration and gauge as :func:`min_region_exhaustive` (offsets drawn
    from ``0 .. N``; ``0`` pinned into the first direction).  The conjecture is a
    statement about *every* minimal configuration, so a downstream check should
    inspect each returned config (de-duplicating by combinatorial type as
    desired).  Translates/relabellings of one type recur here; the pool must be
    rich enough to realise every minimal combinatorial type (true for the small
    instances where this is feasible).
    """
    normals, max_counts = merge_normals_nd(normals, max_counts)
    N = sum(max_counts)
    if pool is None:
        pool = list(range(N + 1))
    pool_f = [Fraction(int(p)) for p in pool]
    zero = Fraction(0)

    choices: List[List[Tuple[Fraction, ...]]] = []
    total = 1
    for idx, a in enumerate(max_counts):
        if a > len(pool_f):
            raise ValueError(f"pool too small for multiplicity {a}")
        combos = list(itertools.combinations(pool_f, a))
        if idx == 0 and zero in pool_f:
            combos = [c for c in combos if zero in c]
        choices.append(combos)
        total *= len(combos)
    if total > max_configs:
        raise ValueError(
            f"{total} configurations exceed max_configs={max_configs}; "
            "increase max_configs or shrink the instance/pool"
        )

    best: Optional[int] = None
    configs: List[Tuple[Tuple[Fraction, ...], ...]] = []
    for combo in itertools.product(*choices):
        offsets = tuple(tuple(c) for c in combo)
        r = region_count_nd(normals, offsets)
        if best is None or r < best:
            best = r
            configs = [offsets]
        elif r == best:
            configs.append(offsets)
    return best, list(normals), configs
