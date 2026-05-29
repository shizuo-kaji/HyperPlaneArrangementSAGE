"""Hyperplane arrangement utilities for Sage."""

from .arrangement import HyperplaneArrangement
from .vector_field import VectorField, VectorFieldModule
from .tangential_field import ConvexPolygonFlow, Vortex
from .minimal_region import GreedyCutAllSolver, CppGreedyCutAllSolver, Solution

__all__ = [
    'HyperplaneArrangement',
    'VectorField',
    'VectorFieldModule',
    'ConvexPolygonFlow',
    'Vortex',
    'GreedyCutAllSolver',
    'CppGreedyCutAllSolver',
    'Solution',
]
