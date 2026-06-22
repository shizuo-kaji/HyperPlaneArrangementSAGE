# Notebook Inventory

## Table of Contents

- [Canonical Notebooks](#canonical-notebooks)
- [File Consolidation](#file-consolidation)
- [Archived Experiments](#archived-experiments)
- [Shared Helpers](#shared-helpers)
- [Maintenance Notes](#maintenance-notes)

Use the SageMath kernel unless a notebook says otherwise. Notebooks should stay
as demonstrations or bounded experiments; reusable helpers belong in
`src/hyperplane_arrangements/`.

## Canonical Notebooks

- `Library_Demo.ipynb` — package library examples for Coxeter and affine
  arrangements.
- `LogarithmicVectorFieldsOfArrangements.ipynb` — broad reference notebook for
  logarithmic vector fields, restrictions, deletions, and finite-field examples.
- `Saito_criterion_examples.ipynb` — canonical Saito/generalized-Saito examples
  and stress tests.
- `B2_multi_arrangement.ipynb` — focused `B2` multi-arrangement examples.
- `constructive_closure.ipynb` — short constructive-closure visualization demo.
- `MinimalRegionPlaneCut.ipynb` — generated minimal-region experiment notebook.
  Regenerate from the root `build_nb.py` if that generator is restored.
- `vector_field_reconstruction.ipynb` — 2-D vector-field reconstruction examples.
- `vector_field_reconstruction_3d.ipynb` — 3-D vector-field reconstruction and
  PyVista visualization examples.
- `Examples_NTF-2.ipynb` — paper-oriented NTF examples. Keep bounded; move
  scratch searches to `experiments/`.

## File Consolidation

- `experiment_saito.ipynb` was moved to `experiments/` because
  `Saito_criterion_examples.ipynb` is the canonical version of that line of
  experiments.
- `vector_field_reconstruction.ipynb` and `vector_field_reconstruction_3d.ipynb`
  are related, but remain separate for now because the 3-D notebook depends on
  PyVista/SciPy visualization machinery while the 2-D notebook is lighter and
  Matplotlib-only.
- `constructive_closure.ipynb` remains separate as a short visualization demo.
  Its reusable plotting helpers now live in
  `hyperplane_arrangements.arrangement_plotting`.
- `Examples_NTF-2.ipynb` overlaps thematically with
  `Saito_criterion_examples.ipynb`, but it is paper-example oriented. If it grows
  more diagnostic code, move that code into the Saito notebook or package tests
  and keep this notebook as a compact gallery.
- `MinimalRegionPlaneCut.ipynb` should stay separate because it is generated and
  tied to bounded minimal-region sweeps.

## Archived Experiments

- `experiments/experiment_saito.ipynb` — scratch Saito investigation superseded
  by `Saito_criterion_examples.ipynb`.

## Shared Helpers

The vector-field notebooks now import common numerical and plotting helpers from
`hyperplane_arrangements.field_tools` instead of carrying local copies.

Constructive-closure plotting helpers live in
`hyperplane_arrangements.arrangement_plotting`.

## Maintenance Notes

- Avoid section numbers in notebook headings; future reordering should not force
  renumbering.
- Keep broad reusable functions in `src/hyperplane_arrangements/`.
- Keep throwaway sweeps and exploratory copies in `experiments/`.
