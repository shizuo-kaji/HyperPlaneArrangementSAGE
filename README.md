# Logarithmic Derivation Module of Hyperplane Arrangement

A Sage package for working with logarithmic vector fields and hyperplane arrangements, with applications to vector field reconstruction.

## Authors
Developed by **Junyan Chu** and **Shizuo Kaji**

JC was supported by the China Scholarship Council.

## Overview

This repository provides tools for:

- Analyzing hyperplane arrangements with a particular focus on logarithmic derivation modules
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

### Core Package

```python
from hyperplane_arrangements import *
A = HyperPlaneArr([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
print(gendic(A.minimal_generators()))
print(A.free_resolution())
A.plot(A.minimal_generators()[1], xlim=(-2, 2), ylim=(-2, 2))
```
For more details, see the Jupyter Notebooks below:

- **`src/hyperplane_arrangements/arrangement.py`**
  - Implementation of `HyperplaneArrangement` class

- **`src/hyperplane_arrangements/tangential_field.py`**
  - Synthetic `ConvexPolygonFlow` generator
  - Produces tangential vector-field samples, vorticity, and divergence

## Quick Start Example

See Jupyter Notebooks.

- **`LogarithmicVectorFieldsOfArrangements.ipynb`**
  - Demonstrates usage of the logarithmic derivation module functions

- **`Examples_NTF-2.ipynb`**
  - Examples in Chu, Junyan. Free resolution of the logarithmic derivation modules of close to free arrangements. J Algebr Comb 61, 26 (2025).
  https://doi.org/10.1007/s10801-025-01394-7.

- **`vector_field_reconstruction.ipynb`**
  - Demonstrates vector field reconstruction from synthetic data
  - Junyan Chu, Shizuo Kaji: Polynomial Interpolation of a Vector Field on a Convex Polygonal Domain, preprint
