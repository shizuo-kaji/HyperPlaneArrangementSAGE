"""Structured diagnostics for Saito-type criteria."""

from __future__ import annotations

import itertools
from typing import Iterable, Optional


def generator_list(A, generators=None):
    """Return the candidate generators as a plain list."""
    return list(A.minimal_generators() if generators is None else generators)


def projective_dimension_D(A) -> int:
    """Return ``pd_S D(A)`` from the minimal free resolution."""
    return A.free_resolution()._length - 1


def scaled_minor_subset_diagnostics(A, generators=None, *, verify: bool = False):
    """Return scaled-minor diagnostics for all ``ell+1`` generator sublists."""
    G = generator_list(A, generators)
    ell = A.n
    p = len(G)
    ideal = A.scaled_minor_ideal(G)
    height = A.scaled_minor_ideal_height(G)

    subset_results = []
    if p >= ell + 1:
        for subset in itertools.combinations(range(p), ell + 1):
            sub_generators = [G[j] for j in subset]
            result = A.check_generalized_saito(generators=sub_generators, verify=verify, verbose=False)
            subset_results.append((subset, result))

    height_counts = {}
    for _, result in subset_results:
        height_counts[result["height"]] = height_counts.get(result["height"], 0) + 1

    return {
        "num_generators": p,
        "dimension": ell,
        "ideal": ideal,
        "height": height,
        "proper": not ideal.is_one(),
        "subset_results": subset_results,
        "height_counts": height_counts,
        "num_subsets": len(subset_results),
        "num_criterion": sum(1 for _, result in subset_results if result["criterion_applies"]),
        "num_predicts_minimal_spog": sum(1 for _, result in subset_results if result["predicts_minimal_spog"]),
        "num_actually_generates": sum(
            1 for _, result in subset_results if result.get("actually_generates")
        ),
    }


def height_bound_diagnostic(A, generators=None):
    """Return data for the conjectural bound ``ht I_A(G) >= pd_S D(A) + 2``."""
    G = generator_list(A, generators)
    ideal = A.scaled_minor_ideal(G)
    is_proper = not ideal.is_one()
    height = A.scaled_minor_ideal_height(G)
    pd = projective_dimension_D(A)
    bound = pd + 2
    ok = True if not is_proper else height >= bound
    return {
        "projective_dimension": pd,
        "height": height,
        "bound": bound,
        "is_proper": is_proper,
        "ok": ok,
        "num_generators": len(G),
        "degrees": A.degrees(),
    }
