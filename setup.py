"""Setup script for the Hyperplane Arrangements Sage package."""
from pathlib import Path

from sage_setup import setup
from setuptools import find_packages

ROOT = Path(__file__).parent

setup(
    name="hyperplane-arrangements",
    version="0.1.0",
    description="Logarithmic vector fields for convex hyperplane arrangements",
    author="Maple Hyperplane Team",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.11",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)
