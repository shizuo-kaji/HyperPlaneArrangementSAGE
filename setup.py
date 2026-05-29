"""Setup script for the Hyperplane Arrangements Sage package."""
import subprocess
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

ROOT = Path(__file__).parent

class CustomBuildPy(build_py):
    def run(self):
        # Try building the C++ solver optionally
        cpp_dir = ROOT / "src" / "hyperplane_arrangements" / "cpp" / "minimal_region"
        if (cpp_dir / "Makefile").exists():
            try:
                print("Attempting to compile optional C++ solver...")
                subprocess.run(["make", "-C", str(cpp_dir)], check=True)
                print("Successfully compiled C++ solver.")
            except Exception as e:
                print(f"Warning: Failed to compile optional C++ solver: {e}")
        
        # Run normal build
        super().run()

setup(
    name="hyperplane-arrangements",
    version="0.1.0",
    description="Logarithmic vector fields for convex hyperplane arrangements",
    author="Maple Hyperplane Team",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"hyperplane_arrangements": ["cpp/minimal_region/solver", "cpp/minimal_region/Makefile", "cpp/minimal_region/*.cpp"]},
    include_package_data=True,
    python_requires=">=3.11",
    cmdclass={"build_py": CustomBuildPy},
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
)
