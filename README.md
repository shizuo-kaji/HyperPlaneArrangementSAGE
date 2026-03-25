# Logarithmic Derivation Module of Hyperplane Arrangement

A Sage package for logarithmic vector fields and hyperplane arrangements, with applications to vector field reconstruction.

## Authors
Developed by **Junyan Chu** and **Shizuo Kaji**

JC was supported by the China Scholarship Council.

## Overview

This repository provides tools for:

- Analyzing hyperplane arrangements with a particular focus on logarithmic derivation modules
- Working with homogeneous bounded-degree modules of polynomial vector fields
- Fitting polynomial vector fields to velocity and vorticity data on polyhedral domains using least-squares methods

## Prerequisites

To use this module, you need to have access to SageMath. You can:
1. [Install SageMath locally](https://www.sagemath.org/download.html) on your computer.
2. Use an online service such as [CoCalc](https://cocalc.com/), which provides an environment to run SageMath.

## Installation

Install the package into your Sage environment using:

```bash
sage -pip install -e .
```

## Repository Layout

### Core Package `src/hyperplane_arrangements/`

The library employs an object-oriented design handling vector fields independently from arrangements:

- **`arrangement.py`**: Contains the core `HyperplaneArrangement` class.
- **`vector_field.py`**: Houses the `VectorField` and `VectorFieldModule` classes, encapsulating differential operations (div, rot, laplacian) and subspace operations (graded components, free resolutions, dehomogenization).
- **`fit.py`**: Dedicated module and algorithms for fitting vector fields and vorticities against observations.
- **`utils.py`**: Mathematical helpers and utility functions.
- **`tangential_field.py`**: Synthetic `ConvexPolygonFlow` generator for tangential vector-field samples.

Basic Usage:
```python
from hyperplane_arrangements import *
A = HyperPlaneArr([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

# Get the vector field module containing minimal generators
mod = A.minimal_generators()

# Inspect degrees and free resolution
print(mod.gendic())
print(A.free_resolution())

# Work natively with VectorField objects
A.plot(mod.gens[1], xlim=(-2, 2), ylim=(-2, 2))
```

## Important: Jupyter Kernel

> **All notebooks in this repository must be run with the SageMath kernel, not a plain Python kernel.**

When opening a notebook in JupyterLab / VS Code, make sure the kernel is set to **SageMath** (e.g. `SageMath 10.x`).
If you see `ModuleNotFoundError: No module named 'sage'`, switch the kernel to SageMath.

- **JupyterLab**: *Kernel → Change Kernel → SageMath*
- **VS Code**: click the kernel name in the top-right corner of the notebook and select *SageMath*

## Quick Start Example

See Jupyter Notebooks.

- **`LogarithmicVectorFieldsOfArrangements.ipynb`**
  - Demonstrates usage of the logarithmic derivation module functions

- **`Examples_NTF-2.ipynb`**
  - Examples in Chu, Junyan. Free resolution of the logarithmic derivation modules of close to free arrangements. J Algebr Comb 61, 26 (2025).
  https://doi.org/10.1007/s10801-025-01394-7.

- **`vector_field_reconstruction.ipynb`**
  - Demonstrates vector field reconstruction from synthetic data
  - Junyan Chu, Shizuo Kaji: Polynomial Interpolation of a Vector Field on a Convex Polygonal Domain, arXiv:2602.01803
