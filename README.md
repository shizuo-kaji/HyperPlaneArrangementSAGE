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

To use this module, you need SageMath with Jupyter support.

### macOS (recommended: Sage binary app)

1. Download SageMath for macOS from the [official download page](https://www.sagemath.org/download.html).
2. Open the downloaded `.dmg` and drag the Sage app to `Applications`.
3. Start Sage once from `Applications` (this completes first-run setup).
4. Open Terminal and check Sage is available:

```bash
/Applications/SageMath.app/Contents/Resources/sage/sage --version
```

5. Install/enable Jupyter inside Sage's Python environment:

```bash
sage -pip install --upgrade jupyterlab notebook ipykernel
sage --python -m ipykernel install --user --name sagemath --display-name "SageMath"
```

6. Launch Jupyter with Sage from the repository root:

```bash
cd /path/to/HyperplaneArrangement
sage -n jupyter
```

### Windows (recommended: WSL2 + Linux Sage)

SageMath is most reliable on Windows via WSL2.

1. Install WSL2 (PowerShell as Administrator):

```powershell
wsl --install
```

2. Reboot if prompted, then open Ubuntu from the Start menu and create your Linux user.
3. In Ubuntu, install Sage and Jupyter:

```bash
sudo apt update
sudo apt install -y sagemath python3-pip
sage -pip install --upgrade jupyterlab notebook ipykernel
sage --python -m ipykernel install --user --name sagemath --display-name "SageMath"
```

4. Verify installation:

```bash
sage --version
```

5. Start Jupyter from the repository root:

```bash
cd /path/to/HyperplaneArrangement
sage -n jupyter --no-browser --ip=0.0.0.0 --port=8888
```

Then open the shown URL in your Windows browser.

### Verify notebook kernel

In JupyterLab or VS Code, select the kernel named **SageMath** for this project notebooks.
If you get `ModuleNotFoundError: No module named 'sage'`, the notebook is running on the wrong kernel.

## Development Installation

Install the package into the active Sage environment:

```bash
sage -pip install -e .
```

The notebooks in `notebooks/` also add the repository `src/` directory to `sys.path` automatically when run from inside the repository. The editable install is still recommended for normal development and for opening notebooks from external tools.

## Repository Layout

- `src/hyperplane_arrangements/`: Sage package source.
  - `arrangement.py`: `HyperplaneArrangement` core logic.
  - `vector_field.py`: `VectorField` and `VectorFieldModule` abstractions for derivatives, divergence, rotation, Laplacian, graded components, dehomogenization, and module operations.
  - `fit.py`: vector-field and vorticity fitting routines.
  - `utils.py`: helper functions without class-level dependencies.
  - `tangential_field.py`: synthetic `ConvexPolygonFlow` and tangential field generation.
  - `minimal_region.py`: minimal region plane cut search and generator logic.
    - An optional C++ solver is provided in `src/hyperplane_arrangements/cpp/minimal_region/`. It is automatically compiled during package installation if a C++ compiler (`make` and `c++`) is present. The `CppGreedyCutAllSolver` wrapper seamlessly detects and uses it.
- `notebooks/`: active Jupyter notebooks.
- `tests/`: pytest suite for the Sage package.
- `paper/`: LaTeX manuscript and bibliography.
- `old/`: archived notebooks and historical experiment files.

Basic usage:

```python
from hyperplane_arrangements import HyperplaneArrangement

A = HyperplaneArrangement([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

# Get the vector field module containing minimal generators
mod = A.minimal_generators()

# Inspect degrees and free resolution
print(mod.gendic())
print(A.free_resolution())

# Work natively with VectorField objects
A.plot(mod.gens[1], xlim=(-2, 2), ylim=(-2, 2))
```

The legacy alias `HyperPlaneArr` remains available for older notebooks.

## Testing

Inside a Sage-enabled environment, run:

```bash
PYTHONPATH=src:$PYTHONPATH pytest tests/
```

## Important: Jupyter Kernel

> **All notebooks in this repository must be run with the SageMath kernel, not a plain Python kernel.**

When opening a notebook in JupyterLab / VS Code, make sure the kernel is set to **SageMath** (e.g. `SageMath 10.x`).
If you see `ModuleNotFoundError: No module named 'sage'`, switch the kernel to SageMath.

- **JupyterLab**: *Kernel → Change Kernel → SageMath*
- **VS Code**: click the kernel name in the top-right corner of the notebook and select *SageMath*

## Quick Start Example

See the active notebooks under `notebooks/`:

- **`notebooks/LogarithmicVectorFieldsOfArrangements.ipynb`**
  - Demonstrates usage of the logarithmic derivation module functions

- **`notebooks/Examples_NTF-2.ipynb`**
  - Examples in Chu, Junyan. Free resolution of the logarithmic derivation modules of close to free arrangements. J Algebr Comb 61, 26 (2025).
  https://doi.org/10.1007/s10801-025-01394-7.

- **`notebooks/vector_field_reconstruction.ipynb`**
  - Demonstrates vector field reconstruction from synthetic data
  - Junyan Chu, Shizuo Kaji: Polynomial Interpolation of a Vector Field on a Convex Polygonal Domain, arXiv:2602.01803

- **`notebooks/vector_field_reconstruction_3d.ipynb`**
  - Extends the reconstruction workflow to three-dimensional examples.

- **`notebooks/Saito_criterion_examples.ipynb`**
  - Integrated generalized Saito criterion examples, including generic determinant-ideal experiments.

- **`notebooks/B2_multi_arrangement.ipynb`**
  - Multi-arrangement examples.

- **`notebooks/constructive_closure_demo.ipynb`**
  - Constructive closure examples and visualizations.

- **`notebooks/MinimalRegionPlaneCut.ipynb`**
  - Minimal region plane cut solver.
  - Generates exact reachable count tuples for minimal regions.
