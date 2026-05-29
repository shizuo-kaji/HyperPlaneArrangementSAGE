from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Dict, Iterable, List, Optional, Set, Tuple
import csv
import io
import json
import matplotlib.pyplot as plt

Point = Tuple[Fraction, Fraction]
Normal = Tuple[int, int]

ZERO = Fraction(0)
ORIGIN: Point = (ZERO, ZERO)


def normalize_normal(nx: int, ny: int) -> Normal:
    if nx == 0 and ny == 0:
        raise ValueError("Normal (0,0) is invalid")
    g = gcd(abs(nx), abs(ny))
    nx //= g
    ny //= g
    # Canonical sign: first non-zero positive
    if nx < 0 or (nx == 0 and ny < 0):
        nx, ny = -nx, -ny
    return nx, ny


def merge_normals(
    normals: Iterable[Tuple[int, int]],
    max_counts: Iterable[int],
) -> Tuple[List[Normal], List[int]]:
    normals = list(normals)
    max_counts = list(max_counts)
    if len(normals) != len(max_counts):
        raise ValueError("normals and max_counts must have same length")

    agg: Dict[Normal, int] = {}
    for (nx, ny), max_c in zip(normals, max_counts):
        nn = normalize_normal(nx, ny)
        agg[nn] = agg.get(nn, 0) + max_c

    merged_normals = list(agg.keys())
    merged_max = [agg[n] for n in merged_normals]
    return merged_normals, merged_max


def intersection(n1: Normal, c1: Fraction, n2: Normal, c2: Fraction) -> Optional[Point]:
    a1, b1 = n1
    a2, b2 = n2
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None
    x = Fraction(c1 * b2 - c2 * b1, det)
    y = Fraction(a1 * c2 - a2 * c1, det)
    return (x, y)


def dot(n: Normal, p: Point) -> Fraction:
    a, b = n
    x, y = p
    return a * x + b * y


@dataclass(frozen=True)
class Solution:
    regions: int
    normals: Tuple[Normal, ...]
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...]
    seed_points: Tuple[Point, ...]
    generators_by_dir: Tuple[Tuple[Fraction, ...], ...]


class GreedyCutAllSolver:
    def __init__(
        self,
        normals: Iterable[Tuple[int, int]],
        max_counts: Iterable[int],
    ) -> None:
        normals, max_counts = merge_normals(normals, max_counts)
        self.normals: Tuple[Normal, ...] = tuple(normals)
        self.max_counts: Tuple[int, ...] = tuple(max_counts)
        self.m = len(self.normals)

        if (0, 1) not in self.normals:
            raise ValueError("normals must include (0, 1)")

        self.best_for_counts: Dict[
            Tuple[int, ...],
            Tuple[int, Tuple[frozenset[Fraction], ...], frozenset[Point]],
        ] = {}
        self.visited: Set[Tuple[Tuple[frozenset[Fraction], ...], frozenset[Point]]] = set()
        self.initial_seed: frozenset[Point] = frozenset({ORIGIN})

        # Hot-loop caches: avoid recomputing repeated rational arithmetic.
        self._dot_cache: Dict[Tuple[int, Point], Fraction] = {}
        self._intersection_cache: Dict[Tuple[int, Fraction, int, Fraction], Optional[Point]] = {}
        self._pair_coeffs: Dict[Tuple[int, int], Tuple[int, int, int, int, int]] = {}
        for i, (a1, b1) in enumerate(self.normals):
            for j in range(i + 1, self.m):
                a2, b2 = self.normals[j]
                det = a1 * b2 - a2 * b1
                self._pair_coeffs[(i, j)] = (a1, b1, a2, b2, det)

    def solve_all(self) -> Dict[Tuple[int, ...], Solution]:
        empty_lines = tuple(frozenset() for _ in range(self.m))
        self._dfs(empty_lines, self.initial_seed, self.initial_seed, 1)
        result: Dict[Tuple[int, ...], Solution] = {}
        for counts, (regions, lines, seeds) in self.best_for_counts.items():
            lines_sorted = tuple(tuple(sorted(s)) for s in lines)
            seed_points = tuple(sorted(p for p in seeds if p not in self.initial_seed))
            generators_by_dir = find_generators(self.normals, lines_sorted)
            result[counts] = Solution(
                int(regions),
                self.normals,
                lines_sorted,
                seed_points,
                generators_by_dir,
            )
        return result

    def _record(
        self,
        counts: Tuple[int, ...],
        regions: int,
        lines: Tuple[frozenset[Fraction], ...],
        seeds: frozenset[Point],
    ) -> None:
        current = self.best_for_counts.get(counts)
        if current is None or regions < current[0]:
            self.best_for_counts[counts] = (regions, lines, seeds)

    def _dot_at(self, i: int, p: Point) -> Fraction:
        key = (i, p)
        if key in self._dot_cache:
            return self._dot_cache[key]
        a, b = self.normals[i]
        x, y = p
        val = a * x + b * y
        self._dot_cache[key] = val
        return val

    def _intersection_at(self, i: int, c: Fraction, j: int, c2: Fraction) -> Optional[Point]:
        if i < j:
            key = (i, c, j, c2)
            ii, jj = i, j
            cc, cc2 = c, c2
        else:
            key = (j, c2, i, c)
            ii, jj = j, i
            cc, cc2 = c2, c

        if key in self._intersection_cache:
            return self._intersection_cache[key]

        a1, b1, a2, b2, det = self._pair_coeffs[(ii, jj)]
        if det == 0:
            self._intersection_cache[key] = None
            return None

        x = Fraction(cc * b2 - cc2 * b1, det)
        y = Fraction(a1 * cc2 - a2 * cc, det)
        pt = (x, y)
        self._intersection_cache[key] = pt
        return pt

    def _next_seed_point(
        self,
        lines: Tuple[frozenset[Fraction], ...],
        counts: Tuple[int, ...],
        seeds: frozenset[Point],
    ) -> Optional[Point]:
        active_dirs = [i for i in range(self.m) if counts[i] < self.max_counts[i]]
        if not active_dirs:
            return None

        used_seeds: Set[Point] = set(seeds)
        existing_intersections: Set[Point] = set()
        for i in range(self.m):
            if not lines[i]:
                continue
            for j in range(i + 1, self.m):
                if not lines[j]:
                    continue
                for c in lines[i]:
                    for c2 in lines[j]:
                        pt = self._intersection_at(i, c, j, c2)
                        if pt is not None:
                            existing_intersections.add(pt)
        blocked_offsets: Dict[int, Set[Fraction]] = {
            i: {self._dot_at(i, p) for p in existing_intersections}
            for i in active_dirs
        }

        radius = 0
        while True:
            candidates: List[Point] = []
            if radius == 0:
                candidates.append(ORIGIN)
            else:
                for x in range(-radius, radius + 1):
                    candidates.append((Fraction(x), Fraction(-radius)))
                    candidates.append((Fraction(x), Fraction(radius)))
                for y in range(-radius + 1, radius):
                    candidates.append((Fraction(-radius), Fraction(y)))
                    candidates.append((Fraction(radius), Fraction(y)))

            for candidate in candidates:
                if candidate in used_seeds:
                    continue
                if any(self._dot_at(i, candidate) in lines[i] for i in range(self.m)):
                    continue
                if all(self._dot_at(i, candidate) not in blocked_offsets[i] for i in active_dirs):
                    return candidate

            radius += 1

    def _dfs(
        self,
        lines: Tuple[frozenset[Fraction], ...],
        points: frozenset[Point],
        seeds: frozenset[Point],
        regions: int,
    ) -> None:
        state_key = (lines, seeds)
        if state_key in self.visited:
            return
        self.visited.add(state_key)

        counts = tuple(len(s) for s in lines)
        self._record(counts, regions, lines, seeds)

        active_dirs = [i for i in range(self.m) if counts[i] < self.max_counts[i]]
        if not active_dirs:
            return

        # Collect unique candidate moves.
        candidates: Dict[Tuple[int, Fraction], Tuple[int, frozenset[Point]]] = {}
        for i in active_dirs:
            existing_offsets = lines[i]
            candidate_offsets: Set[Fraction] = set()

            for p in points:
                c = self._dot_at(i, p)
                if c not in existing_offsets:
                    candidate_offsets.add(c)

            if not candidate_offsets:
                continue

            other_lines = [
                (j, offsets)
                for j, offsets in enumerate(lines)
                if j != i and offsets
            ]

            for c in candidate_offsets:
                new_points: Set[Point] = set()
                for j, offsets in other_lines:
                    for c2 in offsets:
                        pt = self._intersection_at(i, c, j, c2)
                        if pt is not None:
                            new_points.add(pt)
                candidates[(i, c)] = (len(new_points) + 1, frozenset(new_points))

        moves = [
            (i, c, delta, new_points)
            for (i, c), (delta, new_points) in candidates.items()
        ]
        moves.sort(key=lambda x: x[2])  # greedy: smallest increase first

        if not moves:
            seed = self._next_seed_point(lines, counts, seeds)
            if seed is None or seed in points:
                return
            seed_set = frozenset({seed})
            self._dfs(lines, points | seed_set, seeds | seed_set, regions)
            return

        for i, c, delta, new_points in moves:
            new_lines = list(lines)
            new_lines[i] = lines[i] | frozenset({c})
            self._dfs(tuple(new_lines), points | new_points, seeds, regions + delta)


# --------- Generator helpers ---------

def _line_entries_from_arrangement(
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> Tuple[Tuple[Tuple[int, Fraction], ...], Dict[Tuple[int, Fraction], int]]:
    line_entries: List[Tuple[int, Fraction]] = []
    index_by_line: Dict[Tuple[int, Fraction], int] = {}
    for i, offsets in enumerate(lines_by_dir):
        for c in sorted(offsets):
            line = (i, c)
            if line in index_by_line:
                raise ValueError("duplicate lines are not allowed")
            index_by_line[line] = len(line_entries)
            line_entries.append(line)
    return tuple(line_entries), index_by_line


def _points_with_masks(
    normals: Tuple[Normal, ...],
    line_entries: Tuple[Tuple[int, Fraction], ...],
) -> Dict[Point, int]:
    """Map each intersection point to the bitmask of lines (indices into
    ``line_entries``) passing through it.

    Coincident crossings are merged exactly via the rational ``Point`` key, so a
    point where ``k`` lines meet carries a mask with ``k`` bits set (rather than
    ``C(k, 2)`` separate pair entries).  This is the single source of truth for
    intersection data; the other helpers consume it.
    """
    point_to_mask: Dict[Point, int] = {}
    for idx1, (i, c1) in enumerate(line_entries):
        for idx2 in range(idx1 + 1, len(line_entries)):
            j, c2 = line_entries[idx2]
            if i == j:
                continue
            pt = intersection(normals[i], c1, normals[j], c2)
            if pt is None:
                continue
            point_to_mask[pt] = point_to_mask.get(pt, 0) | (1 << idx1) | (1 << idx2)
    return point_to_mask


def _point_masks_from_arrangement(
    normals: Tuple[Normal, ...],
    line_entries: Tuple[Tuple[int, Fraction], ...],
) -> Tuple[int, ...]:
    return tuple(_points_with_masks(normals, line_entries).values())


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _closure_mask(initial_mask: int, point_masks: Tuple[int, ...]) -> int:
    active = initial_mask
    changed = True
    while changed:
        changed = False
        for point_mask in point_masks:
            if _popcount(active & point_mask) < 2:
                continue
            expanded = active | point_mask
            if expanded != active:
                active = expanded
                changed = True
    return active


def _mask_to_lines_by_dir(
    mask: int,
    line_entries: Tuple[Tuple[int, Fraction], ...],
    num_dirs: int,
) -> Tuple[Tuple[Fraction, ...], ...]:
    grouped: List[List[Fraction]] = [[] for _ in range(num_dirs)]
    for idx, (i, c) in enumerate(line_entries):
        if mask & (1 << idx):
            grouped[i].append(c)
    return tuple(tuple(group) for group in grouped)


def generate_subarrangement(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
    generators_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> Tuple[Tuple[Fraction, ...], ...]:
    if len(normals) != len(lines_by_dir):
        raise ValueError("normals and lines_by_dir must have the same length")
    if len(generators_by_dir) != len(lines_by_dir):
        raise ValueError("generators_by_dir and lines_by_dir must have the same length")

    line_entries, index_by_line = _line_entries_from_arrangement(lines_by_dir)
    point_masks = _point_masks_from_arrangement(normals, line_entries)
    initial_mask = 0
    for i, offsets in enumerate(generators_by_dir):
        for c in offsets:
            idx = index_by_line.get((i, c))
            if idx is None:
                raise ValueError("generators_by_dir must be a subset of lines_by_dir")
            initial_mask |= 1 << idx

    closed_mask = _closure_mask(initial_mask, point_masks)
    return _mask_to_lines_by_dir(closed_mask, line_entries, len(lines_by_dir))


def find_generators(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> Tuple[Tuple[Fraction, ...], ...]:
    if len(normals) != len(lines_by_dir):
        raise ValueError("normals and lines_by_dir must have the same length")

    line_entries, _ = _line_entries_from_arrangement(lines_by_dir)
    if not line_entries:
        return tuple(tuple() for _ in range(len(lines_by_dir)))

    point_masks = _point_masks_from_arrangement(normals, line_entries)
    full_mask = (1 << len(line_entries)) - 1
    closure_cache: Dict[int, int] = {}

    def closure(mask: int) -> int:
        cached = closure_cache.get(mask)
        if cached is not None:
            return cached
        closed_mask = _closure_mask(mask, point_masks)
        closure_cache[mask] = closed_mask
        return closed_mask

    # Breadth-first search on closed subarrangements gives an exact minimal generator set.
    queue = deque([0])
    parent: Dict[int, Tuple[Optional[int], Optional[int]]] = {0: (None, None)}
    while queue and full_mask not in parent:
        closed = queue.popleft()
        candidates = [
            idx
            for idx in range(len(line_entries))
            if not (closed & (1 << idx))
        ]
        candidates.sort(
            key=lambda idx: (
                -_popcount(closure(closed | (1 << idx)) ^ closed),
                idx,
            )
        )
        for idx in candidates:
            next_closed = closure(closed | (1 << idx))
            if next_closed in parent:
                continue
            parent[next_closed] = (closed, idx)
            if next_closed == full_mask:
                break
            queue.append(next_closed)

    if full_mask not in parent:
        raise RuntimeError("failed to find generators")

    chosen_mask = 0
    state = full_mask
    while True:
        prev_state, idx = parent[state]
        if prev_state is None or idx is None:
            break
        chosen_mask |= 1 << idx
        state = prev_state
    return _mask_to_lines_by_dir(chosen_mask, line_entries, len(lines_by_dir))


def with_generators(sol: Solution) -> Solution:
    return Solution(
        regions=sol.regions,
        normals=sol.normals,
        lines_by_dir=sol.lines_by_dir,
        seed_points=sol.seed_points,
        generators_by_dir=find_generators(sol.normals, sol.lines_by_dir),
    )


def generator_size(sol: Solution) -> int:
    return sum(len(offsets) for offsets in sol.generators_by_dir)


# --------- Conjecture-test helpers ---------
#
# Combinatorial measurements used to experiment with the conjectures in
# ``doc/conjectures.tex``.  They all act on the same ``(normals, lines_by_dir)``
# representation as :func:`find_generators` and stay pure-Python (Sage-free).
# Labels in the docstrings refer to the LaTeX statements they evaluate.


def chamber_count(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> int:
    r"""Number of chambers of the affine arrangement via Zaslavsky's formula
    (``lem:quasi-saturated``):

    .. math:: \chi(A) = 1 + |A| + \sum_{X \in L_2(A)} (\mathrm{mult}(X) - 1),

    where ``|A|`` is the number of lines and ``mult(X)`` the number of lines
    through the crossing ``X``.  Computed independently of the solver's
    incremental count, so it doubles as a cross-check of ``Solution.regions``.
    """
    line_entries, _ = _line_entries_from_arrangement(lines_by_dir)
    point_masks = _point_masks_from_arrangement(normals, line_entries)
    return 1 + len(line_entries) + sum(_popcount(mask) - 1 for mask in point_masks)


def intersection_multiplicities(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> Dict[int, int]:
    r"""Return ``{k: p_k}`` where ``p_k`` is the number of points at which
    exactly ``k`` lines meet (``k >= 2``)."""
    line_entries, _ = _line_entries_from_arrangement(lines_by_dir)
    point_masks = _point_masks_from_arrangement(normals, line_entries)
    return dict(Counter(_popcount(mask) for mask in point_masks))


def saturated_directions(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> List[int]:
    r"""Indices ``i`` for which the arrangement is *saturated* in direction
    ``normals[i]`` (``def:saturated``): every crossing point lies on one of the
    lines of that direction.

    A direction carrying no line cannot saturate.  When the arrangement has no
    crossing at all the condition is vacuous, so every nonempty direction
    qualifies (the mathematically correct degenerate answer).
    """
    line_entries, _ = _line_entries_from_arrangement(lines_by_dir)
    points = list(_points_with_masks(normals, line_entries).keys())
    result: List[int] = []
    for i, offsets in enumerate(lines_by_dir):
        if not offsets:
            continue
        offset_set = set(offsets)
        if all(dot(normals[i], p) in offset_set for p in points):
            result.append(i)
    return result


def s_invariant(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> int:
    r"""The illegal-move number ``s(A)`` = size of a minimum constructive
    generating subset (``prop:s-equals-illegal``).  Equal to
    :func:`generator_size` of a :class:`Solution` carrying the same lines.
    """
    return sum(len(offsets) for offsets in find_generators(normals, lines_by_dir))


def _det(a: Normal, b: Normal) -> int:
    return a[0] * b[1] - a[1] * b[0]


def is_B2_type(normals: Iterable[Tuple[int, int]]) -> bool:
    r"""Whether the directions form a ``B_2`` system (``def:B2-type``).

    Uses the integer criterion of ``lem:det-B2``:

    .. math:: \det(\alpha_3,\alpha_1)\det(\alpha_4,\alpha_2)
              + \det(\alpha_3,\alpha_2)\det(\alpha_4,\alpha_1) = 0

    for some splitting of the four directions into an axis pair
    ``{alpha_1, alpha_2}`` and a diagonal pair ``{alpha_3, alpha_4}``.  The
    expression is invariant under swapping the two pairs and under swaps within
    a pair, so only the three unordered partitions need to be tested.  Returns
    ``False`` unless there are exactly four distinct directions (integer-only;
    no division or infinite-slope special cases).
    """
    distinct: List[Normal] = []
    seen: Set[Normal] = set()
    for nx, ny in normals:
        nn = normalize_normal(nx, ny)
        if nn not in seen:
            seen.add(nn)
            distinct.append(nn)
    if len(distinct) != 4:
        return False
    a, b, c, d = distinct
    for a1, a2, a3, a4 in ((a, b, c, d), (a, c, b, d), (a, d, b, c)):
        if _det(a3, a1) * _det(a4, a2) + _det(a3, a2) * _det(a4, a1) == 0:
            return True
    return False


def is_quasi_saturated(
    normals: Tuple[Normal, ...],
    lines_by_dir: Tuple[Tuple[Fraction, ...], ...],
) -> bool:
    r"""Whether the arrangement is *quasi-saturated* (``def:quasi-saturated``):
    four ``B_2`` directions with multiplicities ``(2p, 2p, 1, 1)`` (``p >= 2``)
    and crossing profile ``(p_2, p_3, p_4) = (4(p^2 - p + 1), 4p - 3, 1)``.

    (The ``lem:cutoff`` proof sketch writes ``p_2 = 2``; that counts only the
    diagonal double points.  The full count is ``4(p^2 - p + 1)``.)
    """
    counts = sorted(len(offsets) for offsets in lines_by_dir)
    if len(counts) != 4:
        return False
    if counts[0] != 1 or counts[1] != 1 or counts[2] != counts[3]:
        return False
    twop = counts[2]
    if twop < 4 or twop % 2 != 0:
        return False
    if not is_B2_type(normals):
        return False
    p = twop // 2
    expected = {2: 4 * (p * p - p + 1), 3: 4 * p - 3, 4: 1}
    return intersection_multiplicities(normals, lines_by_dir) == expected


def minchamber_n2(m: Tuple[int, int]) -> int:
    r"""Closed form of ``minchamber(P, m)`` for ``|P| = 2`` (``prop:n=2``):
    ``(a_1 + 1)(a_2 + 1)``."""
    a1, a2 = m
    return (a1 + 1) * (a2 + 1)


def minchamber_n3(m: Iterable[int]) -> int:
    r"""Closed form of ``minchamber(P, m)`` for ``|P| = 3`` (``prop:n=3`` /
    ``eq:1``) with ``a_1 = max``:

    .. math:: (a_1+1)(a_2+a_3+1)
              + \big\lfloor \max(-a_1+a_2+a_3,\,1)^2 / 4 \big\rfloor.
    """
    a1, a2, a3 = sorted(m, reverse=True)
    return (a1 + 1) * (a2 + a3 + 1) + (max(-a1 + a2 + a3, 1) ** 2) // 4


def yoshinaga_bound(d1: int, d2: int) -> int:
    r"""Yoshinaga lower bound ``(1 + d_1)(1 + d_2)`` (``cor:yoshinaga``)."""
    return (1 + d1) * (1 + d2)


def saturation_chamber(a: int, N: int) -> int:
    r"""Chamber count of an ``alpha``-saturated arrangement (``prop:saturated-min``):
    ``(a + 1)(N - a + 1)`` with ``a = m(alpha)``."""
    return (a + 1) * (N - a + 1)


def saturation_lower_bound(N: int, n: int) -> Fraction:
    r"""Lower bound on the saturating multiplicity (``corollary:houwa_lowerbound``):
    ``(N - n + 2) / 2`` (generally a half-integer)."""
    return Fraction(N - n + 2, 2)


# --------- Utility helpers ---------

def _fraction_to_text(value: Fraction) -> str:
    return str(value)


def _point_to_texts(point: Point) -> List[str]:
    return [_fraction_to_text(point[0]), _fraction_to_text(point[1])]


def _point_from_texts(values: Iterable[str]) -> Point:
    x_text, y_text = values
    return (Fraction(x_text), Fraction(y_text))


def write_json(path: str, results: Dict[Tuple[int, ...], Solution]) -> None:
    items = sorted(results.items())
    normals: Tuple[Normal, ...] = ()
    if items:
        normals = items[0][1].normals
        for _, sol in items[1:]:
            if sol.normals != normals:
                raise ValueError("All solutions must use the same normals")

    payload = {
        "normals": [list(normal) for normal in normals],
        "results": [
            {
                "counts": list(counts),
                "regions": sol.regions,
                "seed_points": [_point_to_texts(point) for point in sol.seed_points],
                "lines_by_dir": [
                    [_fraction_to_text(offset) for offset in offsets]
                    for offsets in sol.lines_by_dir
                ],
                "generators_by_dir": [
                    [_fraction_to_text(offset) for offset in offsets]
                    for offsets in sol.generators_by_dir
                ],
            }
            for counts, sol in items
        ],
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def read_json(path: str) -> Dict[Tuple[int, ...], Solution]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    normals = tuple(tuple(normalize_normal(int(nx), int(ny))) for nx, ny in payload.get("normals", []))

    results: Dict[Tuple[int, ...], Solution] = {}
    for entry in payload.get("results", []):
        counts = tuple(int(x) for x in entry["counts"])
        lines_by_dir = tuple(
            tuple(Fraction(offset) for offset in offsets)
            for offsets in entry["lines_by_dir"]
        )
        if normals and len(lines_by_dir) != len(normals):
            raise ValueError("lines_by_dir and normals must have the same length")
        seed_points = tuple(_point_from_texts(point) for point in entry.get("seed_points", []))
        generators_raw = entry.get("generators_by_dir")
        if generators_raw is None:
            generators_by_dir = find_generators(normals, lines_by_dir)
        else:
            generators_by_dir = tuple(
                tuple(Fraction(offset) for offset in offsets)
                for offsets in generators_raw
            )
            if len(generators_by_dir) != len(lines_by_dir):
                raise ValueError("generators_by_dir and lines_by_dir must have the same length")
        results[counts] = Solution(
            regions=int(entry["regions"]),
            normals=normals,
            lines_by_dir=lines_by_dir,
            seed_points=seed_points,
            generators_by_dir=generators_by_dir,
        )

    return dict(sorted(results.items()))


def write_csv(path: str, results: Dict[Tuple[int, ...], Solution]) -> None:
    rows_by_n: Dict[int, List[Tuple[Tuple[int, ...], Solution]]] = {}
    for counts, sol in sorted(results.items()):
        if len(counts) != 4:
            raise ValueError("write_csv expects 4-tuples of counts")
        if any(count < 0 for count in counts):
            raise ValueError("write_csv expects positive counts")
        total_count = sum(counts)
        if total_count % 2 == 0:
            n = -1
        else:
            n = (total_count - 1) // 2
        rows_by_n.setdefault(n, []).append((counts, sol))

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow([
        "n",
        "index",
        "total",
        "a1",
        "a2",
        "a3",
        "a4",
        "min_regions",
        "seed_points",
        "expected_regions (n+1)(n+2)",
        "matches_expected",
    ])
    for n in sorted(rows_by_n):
        group = sorted(rows_by_n[n], key=lambda item: item[0])
        total = len(group)
        expected = (n + 1) * (n + 2)
        for index, (counts, sol) in enumerate(group, start=1):
            writer.writerow([
                n,
                index,
                total,
                *counts,
                sol.regions,
                len(sol.seed_points),
                expected,
                "TRUE" if sol.regions == expected else "FALSE",
            ])

    with open(path, "w", encoding="utf-8") as f:
        f.write(buffer.getvalue().rstrip("\n"))


def format_solution(sol: Solution) -> None:
    print(f"Min regions = {sol.regions}")
    print(f"generator size s(A) = {generator_size(sol)}")
    for normal, offsets in zip(sol.normals, sol.generators_by_dir):
        if not offsets:
            continue
        print(f"generator normal={normal}, count={len(offsets)}")
        for c in offsets:
            print(f"  generator c = {c}")
    print(f"seed points added by _next_seed_point = {len(sol.seed_points)}")
    for p in sol.seed_points:
        print(f"  seed = {p}")
    for normal, offsets in zip(sol.normals, sol.lines_by_dir):
        print(f"normal={normal}, count={len(offsets)}")
        for c in offsets:
            print(f"  c = {c}")


def plot_solution(sol: Solution, margin: float = 1.0) -> None:
    intersections: List[Point] = []
    for i in range(len(sol.normals)):
        for c1 in sol.lines_by_dir[i]:
            for j in range(i + 1, len(sol.normals)):
                for c2 in sol.lines_by_dir[j]:
                    pt = intersection(sol.normals[i], c1, sol.normals[j], c2)
                    if pt is not None:
                        intersections.append(pt)

    if intersections:
        xs = [float(p[0]) for p in intersections]
        ys = [float(p[1]) for p in intersections]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x, max_x, min_y, max_y = -1.0, 1.0, -1.0, 1.0

    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    for normal, offsets in zip(sol.normals, sol.lines_by_dir):
        a, b = normal
        for c in offsets:
            c_float = float(c)
            if b != 0:
                x1, x2 = min_x, max_x
                y1 = (c_float - a * x1) / b
                y2 = (c_float - a * x2) / b
                plt.plot([x1, x2], [y1, y2], alpha=0.7)
            else:
                if a == 0:
                    continue
                x = c_float / a
                plt.vlines(x, min_y, max_y, alpha=0.7)

    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.title(f"Min Regions = {sol.regions}")
    plt.grid(True, linestyle=":", alpha=0.3)
    plt.show()


import subprocess
import tempfile
import os
from pathlib import Path

class CppGreedyCutAllSolver:
    def __init__(
        self,
        normals: Iterable[Tuple[int, int]],
        max_counts: Iterable[int],
        threads: int = 1,
        split_depth: int = 0,
        solver_path: Optional[str] = None
    ) -> None:
        self.normals = list(normals)
        self.max_counts = list(max_counts)
        self.threads = threads
        self.split_depth = split_depth

        if solver_path is None:
            # default path relative to this file
            base = Path(__file__).parent.parent.parent / "src" / "hyperplane_arrangements" / "cpp" / "minimal_region"
            self.solver_path = str(base / "solver")
        else:
            self.solver_path = solver_path

    def solve_all(self) -> Dict[Tuple[int, ...], Solution]:
        if not os.path.exists(self.solver_path) or not os.access(self.solver_path, os.X_OK):
            raise RuntimeError(
                f"C++ solver executable not found or not executable at {self.solver_path}. "
                "Please compile it first or use GreedyCutAllSolver (Python) instead."
            )

        input_data = {
            "normals": self.normals,
            "max_counts": self.max_counts,
            "threads": self.threads,
            "split_depth": self.split_depth
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            in_path = os.path.join(temp_dir, "input.json")
            out_path = os.path.join(temp_dir, "output.json")

            with open(in_path, "w") as f:
                json.dump(input_data, f)

            # Run the solver
            subprocess.run([self.solver_path, in_path, out_path], check=True)

            return read_json(out_path)

