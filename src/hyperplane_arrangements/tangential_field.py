"""Synthetic tangential vector fields on convex polygons for testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from matplotlib.path import Path


@dataclass
class Vortex:
    """Simple Rankine vortex used in the synthetic flow."""

    center: np.ndarray
    gamma: float
    core_radius: float


class ConvexPolygonFlow:
    r"""
    Generate a tangential vector field inside a convex polygon via a simple
    vortex-based fluid simulation.

    The simulation consists of advecting a handful of Rankine vortices and
    evaluating the induced velocity field with a slip boundary projection. The
    resulting field is divergence-free (up to numerical projection) and can be
    sampled at arbitrary interior points for testing fitting algorithms.

    INPUT:

    - ``vertices`` -- iterable of ``(x, y)`` coordinates describing a convex
      polygon (counter-clockwise orientation is enforced automatically).
    - ``n_vortices`` -- number of vortices to seed (default: 4).
    - ``gamma_range`` -- tuple ``(min, max)`` controlling vortex strengths.
    - ``core_radius`` -- radius parameter for each Rankine vortex.
    - ``projection_length`` -- characteristic length scale (in the same units as
      the polygon) used when projecting the velocity onto the boundary tangent;
      the projection weight decays exponentially with distance to the boundary.
    - ``n_steps`` -- number of explicit Euler advection steps applied to the
      vortices (default: 25).
    - ``dt`` -- time step for the vortex advection.
    - ``seed`` -- optional integer used to seed NumPy's RNG for reproducible
      samples.

    """

    def __init__(
        self,
        vertices: Sequence[Tuple[float, float]],
        *,
        n_vortices: int = 4,
        gamma_range: Tuple[float, float] = (-3.0, 3.0),
        core_radius: float = 0.05,
        projection_length: float = 0.05,
        n_steps: int = 25,
        dt: float = 0.01,
        seed: Optional[int] = None,
    ) -> None:
        verts = np.array(vertices, dtype=float)
        if verts.shape[0] < 3:
            raise ValueError("need at least three vertices for a polygon")
        if np.linalg.det(np.array([[verts[1, 0] - verts[0, 0], verts[2, 0] - verts[0, 0]],
                                    [verts[1, 1] - verts[0, 1], verts[2, 1] - verts[0, 1]]])) == 0:
            raise ValueError("polygon vertices must be non-collinear")
        self.rng = np.random.default_rng(int(seed) if seed is not None else None)
        self.vertices = self._ensure_ccw(verts)
        self.path = Path(self.vertices)
        self.bounds = np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)])
        self.centroid = self.vertices.mean(axis=0)
        self.projection_length = projection_length
        self.edges = self._precompute_edges(self.vertices)
        self.vortices: List[Vortex] = self._seed_vortices(n_vortices, gamma_range, core_radius)
        self._advect_vortices(n_steps=n_steps, dt=dt)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def velocity(self, point: Tuple[float, float]) -> np.ndarray:
        """Return the tangential velocity at ``point``."""
        point_arr = np.array(point, dtype=float)
        if not self.path.contains_point(point_arr):
            raise ValueError("point lies outside the polygon")
        value = self._raw_velocity(point_arr)
        return self._project_to_tangent(point_arr, value)

    def vorticity(self, point: Tuple[float, float], h: float = 1e-4) -> float:
        """Return the scalar vorticity ``∂v/∂x - ∂u/∂y`` via central differences."""
        x, y = point
        vx_plus = self.velocity(tuple(self._safe_point(np.array([x + h, y]))))
        vx_minus = self.velocity(tuple(self._safe_point(np.array([x - h, y]))))
        vy_plus = self.velocity(tuple(self._safe_point(np.array([x, y + h]))))
        vy_minus = self.velocity(tuple(self._safe_point(np.array([x, y - h]))))
        dv_dx = (vy_plus[1] - vy_minus[1]) / (2 * h)
        du_dy = (vx_plus[0] - vx_minus[0]) / (2 * h)
        return dv_dx - du_dy

    def divergence(self, point: Tuple[float, float], h: float = 1e-4) -> float:
        """Return ``∂u/∂x + ∂v/∂y`` evaluated numerically."""
        x, y = point
        ux_plus = self.velocity(tuple(self._safe_point(np.array([x + h, y]))))
        ux_minus = self.velocity(tuple(self._safe_point(np.array([x - h, y]))))
        uy_plus = self.velocity(tuple(self._safe_point(np.array([x, y + h]))))
        uy_minus = self.velocity(tuple(self._safe_point(np.array([x, y - h]))))
        du_dx = (ux_plus[0] - ux_minus[0]) / (2 * h)
        dv_dy = (uy_plus[1] - uy_minus[1]) / (2 * h)
        return du_dx + dv_dy

    def sample_random_points(self, n: int) -> np.ndarray:
        """Sample ``n`` points uniformly inside the polygon via rejection."""
        samples: List[np.ndarray] = []
        attempts = 0
        max_attempts = 20 * n
        while len(samples) < n and attempts < max_attempts:
            attempts += 1
            candidate = self.rng.uniform(self.bounds[0], self.bounds[1])
            if self.path.contains_point(candidate):
                samples.append(candidate)
        if len(samples) < n:
            raise RuntimeError("failed to sample enough interior points; check polygon size")
        return np.vstack(samples)

    def sample_velocity(self, n: int, noise: float = 0.0) -> Dict[Tuple[float, float], np.ndarray]:
        """Return ``n`` random points mapped to their velocity vectors."""
        points = self.sample_random_points(n)
        result: Dict[Tuple[float, float], np.ndarray] = {}
        for pt in points:
            vel = self.velocity(tuple(pt))
            if noise > 0:
                vel += self.rng.normal(scale=noise, size=2)
            result[(float(pt[0]), float(pt[1]))] = vel
        return result

    def sample_vorticity(self, n: int, noise: float = 0.0) -> Dict[Tuple[float, float], float]:
        """Return ``n`` random points mapped to vorticity values."""
        points = self.sample_random_points(n)
        data: Dict[Tuple[float, float], float] = {}
        for pt in points:
            omega = self.vorticity(tuple(pt))
            if noise > 0:
                omega += float(self.rng.normal(scale=noise))
            data[(float(pt[0]), float(pt[1]))] = omega
        return data

    # ------------------------------------------------------------------
    # simulation helpers
    # ------------------------------------------------------------------
    def _seed_vortices(
        self,
        n_vortices: int,
        gamma_range: Tuple[float, float],
        core_radius: float,
    ) -> List[Vortex]:
        centers = self.sample_random_points(n_vortices)
        gammas = self.rng.uniform(gamma_range[0], gamma_range[1], size=n_vortices)
        return [Vortex(center=center, gamma=float(gamma), core_radius=core_radius)
                for center, gamma in zip(centers, gammas)]

    def _advect_vortices(self, n_steps: int, dt: float) -> None:
        if n_steps <= 0:
            return
        for _ in range(n_steps):
            updates: List[np.ndarray] = []
            for idx, vort in enumerate(self.vortices):
                vel = self._raw_velocity(vort.center, skip_index=idx)
                updates.append(vel)
            for vort, vel in zip(self.vortices, updates):
                new_center = vort.center + dt * vel
                vort.center = self._project_inside(new_center)

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_ccw(verts: np.ndarray) -> np.ndarray:
        area = 0.5 * np.sum(
            verts[:, 0] * np.roll(verts[:, 1], -1) - verts[:, 1] * np.roll(verts[:, 0], -1)
        )
        if area < 0:
            verts = np.flipud(verts)
        return verts

    @staticmethod
    def _precompute_edges(verts: np.ndarray) -> List[Tuple[np.ndarray, float, np.ndarray]]:
        edges: List[Tuple[np.ndarray, float, np.ndarray]] = []
        n = verts.shape[0]
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            tangent = p2 - p1
            length = np.linalg.norm(tangent)
            if length == 0:
                continue
            tangent /= length
            normal = np.array([tangent[1], -tangent[0]])
            c = -normal.dot(p1)
            edges.append((normal, c, tangent))
        return edges

    def _closest_edge(self, point: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        distances = [normal.dot(point) + c for normal, c, _ in self.edges]
        idx = int(np.argmin(np.abs(distances)))
        normal, _, tangent = self.edges[idx]
        return abs(distances[idx]), normal, tangent

    def _project_inside(self, point: np.ndarray) -> np.ndarray:
        if self.path.contains_point(point):
            return point
        direction = self.centroid - point
        for alpha in np.linspace(0.1, 1.0, 10):
            candidate = point + alpha * direction
            if self.path.contains_point(candidate):
                return candidate
        return self.centroid.copy()

    def _safe_point(self, coords: np.ndarray) -> np.ndarray:
        if self.path.contains_point(coords):
            return coords
        return self._project_inside(coords)

    # ------------------------------------------------------------------
    # field evaluation helpers
    # ------------------------------------------------------------------
    def _raw_velocity(self, point: np.ndarray, skip_index: Optional[int] = None) -> np.ndarray:
        vel = np.zeros(2)
        for idx, vort in enumerate(self.vortices):
            if skip_index is not None and idx == skip_index:
                continue
            vel += self._rankine_velocity(point, vort)
        return vel

    @staticmethod
    def _rankine_velocity(point: np.ndarray, vort: Vortex) -> np.ndarray:
        delta = point - vort.center
        r = np.linalg.norm(delta)
        if r < 1e-8:
            delta = np.array([1e-8, 0.0])
            r = 1e-8
        if r <= vort.core_radius:
            vel_theta = vort.gamma * r / (2 * np.pi * vort.core_radius**2)
        else:
            vel_theta = vort.gamma / (2 * np.pi * r)
        tangential = np.array([-delta[1], delta[0]]) / r
        return vel_theta * tangential

    def _project_to_tangent(self, point: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        distance, normal, tangent = self._closest_edge(point)
        if distance <= 0:
            return velocity - np.dot(velocity, normal) * normal
        weight = np.exp(-distance / max(self.projection_length, 1e-6))
        v_tan = velocity - np.dot(velocity, normal) * normal
        return weight * v_tan + (1 - weight) * velocity


__all__ = ['ConvexPolygonFlow', 'Vortex']
