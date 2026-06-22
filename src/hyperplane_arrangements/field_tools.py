"""Numerical helpers for vector-field reconstruction notebooks."""

from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Polygon as PolygonPatch
from matplotlib.path import Path as MplPath


def polygon_outline(vertices: Sequence[Sequence[float]]) -> np.ndarray:
    """Return polygon vertices with the first vertex repeated at the end."""
    poly = np.asarray(vertices, dtype=float)
    return np.vstack([poly, poly[0]])


def plot_polygon(ax, vertices: Sequence[Sequence[float]], **kwargs) -> None:
    """Draw a closed polygon outline on a Matplotlib axis."""
    outline = polygon_outline(vertices)
    style = {"color": "black", "linewidth": 1.5}
    style.update(kwargs)
    ax.plot(outline[:, 0], outline[:, 1], **style)


def point_in_polygon(point: Sequence[float], vertices: Sequence[Sequence[float]]) -> bool:
    """Return whether ``point`` lies inside the polygon."""
    return bool(MplPath(np.asarray(vertices, dtype=float)).contains_point(np.asarray(point, dtype=float)))


def distance_to_segment(point: Sequence[float], v0: Sequence[float], v1: Sequence[float]) -> float:
    """Return the Euclidean distance from ``point`` to the segment ``v0``-``v1``."""
    point = np.asarray(point, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    v1 = np.asarray(v1, dtype=float)
    segment = v1 - v0
    seg_len_sq = float(segment.dot(segment))
    if seg_len_sq == 0:
        return float(np.linalg.norm(point - v0))
    t = float(np.clip((point - v0).dot(segment) / seg_len_sq, 0.0, 1.0))
    projection = v0 + t * segment
    return float(np.linalg.norm(point - projection))


def min_distance_to_boundary(point: Sequence[float], vertices: Sequence[Sequence[float]]) -> float:
    """Return the nearest distance from ``point`` to any polygon edge."""
    vertices = np.asarray(vertices, dtype=float)
    return min(
        distance_to_segment(point, vertices[i], vertices[(i + 1) % len(vertices)])
        for i in range(len(vertices))
    )


def sample_random_points_in_polygon(
    vertices: Sequence[Sequence[float]],
    n_points: int,
    seed: Optional[int] = None,
    min_dist_from_boundary: float = 0.0,
    min_dist_between_points: float = 0.0,
    max_attempts_factor: int = 3000,
) -> np.ndarray:
    """Sample random points inside a polygon by rejection from its bounding box."""
    vertices = np.asarray(vertices, dtype=float)
    rng = np.random.default_rng(seed)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    path = MplPath(vertices)
    points = []
    attempts = 0
    max_attempts = max_attempts_factor * n_points

    while len(points) < n_points and attempts < max_attempts:
        attempts += 1
        candidate = rng.uniform(mins, maxs)
        if not path.contains_point(candidate):
            continue
        if min_dist_from_boundary > 0 and min_distance_to_boundary(candidate, vertices) < min_dist_from_boundary:
            continue
        if min_dist_between_points > 0 and points:
            nearest = min(np.linalg.norm(candidate - np.asarray(existing)) for existing in points)
            if nearest < min_dist_between_points:
                continue
        points.append(candidate)

    if len(points) < n_points:
        raise RuntimeError(f"Only sampled {len(points)} of {n_points} requested points inside the polygon.")
    return np.asarray(points, dtype=float)


def evaluate_planar_field(A, vf, point: Sequence[float]) -> np.ndarray:
    """Evaluate a 2-D affine vector field at ``point``."""
    gens = A.S.gens()
    subs = {gens[0]: float(point[0]), gens[1]: float(point[1])}
    if len(gens) > 2:
        subs[gens[2]] = 1
    return np.array([float(comp.subs(subs)) for comp in vf[:2]], dtype=float)


def evaluate_planar_field_batch(A, vf, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Evaluate a 2-D affine vector field at many points."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return np.array([evaluate_planar_field(A, vf, pt) for pt in pts], dtype=float)


def evaluate_planar_vorticity(A, vf, point: Sequence[float]) -> float:
    """Evaluate scalar vorticity ``dv/dx - du/dy`` for a 2-D vector field."""
    from sage.all import diff

    gens = A.S.gens()
    x, y = gens[0], gens[1]
    subs = {x: float(point[0]), y: float(point[1])}
    if len(gens) > 2:
        subs[gens[2]] = 1
    return float(diff(vf[1], x).subs(subs) - diff(vf[0], y).subs(subs))


def evaluate_planar_vorticity_batch(A, vf, points: Sequence[Sequence[float]]) -> np.ndarray:
    """Evaluate scalar vorticity at many points."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    return np.array([evaluate_planar_vorticity(A, vf, pt) for pt in pts], dtype=float)


def grid_inside_polygon(
    vertices: Sequence[Sequence[float]],
    nx: int = 90,
    ny: int = 90,
    pad_fraction: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``x``/``y`` grid coordinates and a mask for points inside the polygon."""
    vertices = np.asarray(vertices, dtype=float)
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    pad = pad_fraction * (maxs - mins)
    xs = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], nx)
    ys = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], ny)
    xx, yy = np.meshgrid(xs, ys)
    mask = MplPath(vertices).contains_points(np.column_stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    return xs, ys, mask


def field_on_grid(
    evaluator: Callable[[np.ndarray], np.ndarray],
    xs: Sequence[float],
    ys: Sequence[float],
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a planar vector-field callback on a masked grid."""
    xx, yy = np.meshgrid(xs, ys)
    vx = np.full(xx.shape, np.nan, dtype=float)
    vy = np.full(xx.shape, np.nan, dtype=float)
    valid_points = np.column_stack((xx[mask], yy[mask]))
    values = evaluator(valid_points)
    vx[mask] = values[:, 0]
    vy[mask] = values[:, 1]
    speed = np.hypot(vx, vy)
    return vx, vy, speed


def plot_stream_panel(
    ax,
    vertices: Sequence[Sequence[float]],
    xs: Sequence[float],
    ys: Sequence[float],
    mask: np.ndarray,
    evaluator: Callable[[np.ndarray], np.ndarray],
    title: str,
    obs_points: Optional[Sequence[Sequence[float]]] = None,
    obs_vectors: Optional[Sequence[Sequence[float]]] = None,
    density: float = 1.05,
) -> None:
    """Plot clipped streamlines for a planar vector field on a polygon."""
    vx, vy, speed = field_on_grid(evaluator, xs, ys, mask)
    vx_masked = np.ma.array(vx, mask=~mask)
    vy_masked = np.ma.array(vy, mask=~mask)
    speed_masked = np.ma.array(speed, mask=~mask)

    stream = ax.streamplot(
        xs,
        ys,
        vx_masked,
        vy_masked,
        color=speed_masked,
        cmap="viridis",
        density=density,
        linewidth=1.15,
        arrowsize=0.9,
    )
    clip_patch = PolygonPatch(np.asarray(vertices, dtype=float), closed=True, facecolor="none", edgecolor="none")
    ax.add_patch(clip_patch)
    stream.lines.set_clip_path(clip_patch)
    stream.arrows.set_clip_path(clip_patch)

    plot_polygon(ax, vertices)
    if obs_points is not None and obs_vectors is not None:
        pts = np.asarray(obs_points, dtype=float)
        vecs = np.asarray(obs_vectors, dtype=float)
        ax.quiver(pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1], color="r")
        sing_points = pts[(vecs**2).sum(axis=1) < 1e-10]
        if len(sing_points) > 0:
            ax.scatter(sing_points[:, 0], sing_points[:, 1], marker="*", c="r", s=100)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def plot_vorticity_samples(ax, points, values, title: str, cmap: str = "RdBu_r"):
    """Draw colored vorticity samples on a Matplotlib axis."""
    pts = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float)
    vmax = float(np.max(np.abs(values)))
    if vmax == 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    scatter = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        c=values,
        cmap=cmap,
        norm=norm,
        s=60,
        edgecolors="black",
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return scatter


def boundary_samples(vertices: Sequence[Sequence[float]], samples_per_edge: int = 120):
    """Sample boundary points, outward normals, and normalized arclengths."""
    vertices = np.asarray(vertices, dtype=float)
    points = []
    normals = []
    edge_s = []
    edge_lengths = np.linalg.norm(np.roll(vertices, -1, axis=0) - vertices, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge_lengths)))
    total_length = cumulative[-1]

    for i in range(len(vertices)):
        p0 = vertices[i]
        p1 = vertices[(i + 1) % len(vertices)]
        tangent = p1 - p0
        normal = np.array([tangent[1], -tangent[0]], dtype=float)
        normal /= np.linalg.norm(normal)

        for s in np.linspace(0.0, 1.0, samples_per_edge, endpoint=False):
            points.append((1.0 - s) * p0 + s * p1)
            normals.append(normal)
            edge_s.append((cumulative[i] + s * edge_lengths[i]) / total_length)

    return np.asarray(points, dtype=float), np.asarray(normals, dtype=float), np.asarray(edge_s, dtype=float)


def _fixed_total_exponents(total: int, dimension: int):
    if dimension == 1:
        return [(total,)]
    out = []
    for first in range(total + 1):
        for tail in _fixed_total_exponents(total - first, dimension - 1):
            out.append((first, *tail))
    return out


def monomial_exponents(degree: int, dimension: int = 2):
    """Return exponent tuples for monomials of total degree up to ``degree``."""
    if dimension < 1:
        raise ValueError("dimension must be positive")
    out = []
    for d in range(degree + 1):
        out.extend(_fixed_total_exponents(d, dimension))
    return out


def monomial_matrix(points: Sequence[Sequence[float]], degree: int) -> np.ndarray:
    """Evaluate all bivariate monomials up to ``degree`` on ``points``."""
    pts = np.asarray(points, dtype=float)
    return np.column_stack([(pts[:, 0] ** a) * (pts[:, 1] ** b) for a, b in monomial_exponents(degree, 2)])


def velocity_design_matrix(points: Sequence[Sequence[float]], degree: int) -> np.ndarray:
    """Return the least-squares design matrix for unconstrained planar velocity fitting."""
    mon = monomial_matrix(points, degree)
    num_points, num_terms = mon.shape
    design = np.zeros((2 * num_points, 2 * num_terms), dtype=float)
    design[0::2, :num_terms] = mon
    design[1::2, num_terms:] = mon
    return design


def fit_unconstrained_polynomial(points, vectors, degree: int) -> np.ndarray:
    """Fit an unconstrained bivariate polynomial vector field by least squares."""
    design = velocity_design_matrix(points, degree)
    rhs = np.asarray(vectors, dtype=float).reshape(-1)
    return np.linalg.lstsq(design, rhs, rcond=None)[0]


def evaluate_polynomial_field(coeffs, points, degree: int) -> np.ndarray:
    """Evaluate coefficients from ``fit_unconstrained_polynomial`` at points."""
    mon = monomial_matrix(points, degree)
    num_terms = mon.shape[1]
    coeffs = np.asarray(coeffs, dtype=float)
    return np.column_stack((mon @ coeffs[:num_terms], mon @ coeffs[num_terms:]))


def vorticity_on_grid(evaluator: Callable[[np.ndarray], np.ndarray], xs, ys, mask) -> np.ndarray:
    """Evaluate a scalar vorticity callback on a masked grid."""
    xx, yy = np.meshgrid(xs, ys)
    vort = np.full(xx.shape, np.nan, dtype=float)
    valid_points = np.column_stack((xx[mask], yy[mask]))
    vort[mask] = evaluator(valid_points)
    return vort


def plot_vorticity_panel(
    ax,
    vertices,
    xs,
    ys,
    mask,
    evaluator,
    title: str,
    obs_points=None,
    obs_values=None,
    norm=None,
    cmap: str = "RdBu_r",
):
    """Plot scalar vorticity contours clipped to a polygon."""
    vort = vorticity_on_grid(evaluator, xs, ys, mask)
    vort_masked = np.ma.array(vort, mask=~mask)

    if norm is None:
        vmax = np.ma.abs(vort_masked).max()
        if vmax == 0:
            vmax = 1.0
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    cf = ax.contourf(xs, ys, vort_masked, levels=50, cmap=cmap, norm=norm, extend="both")
    ax.contour(xs, ys, vort_masked, levels=10, colors="black", linewidths=0.5, alpha=0.3)
    plot_polygon(ax, vertices)

    if obs_points is not None and obs_values is not None:
        ax.scatter(
            obs_points[:, 0],
            obs_points[:, 1],
            c=obs_values,
            cmap=cmap,
            norm=norm,
            s=40,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    return cf


def sample_points_in_polyhedron(vertices, n: int, seed: Optional[int] = None) -> np.ndarray:
    """Uniformly sample points inside a 3-D convex polyhedron by rejection."""
    from scipy.spatial import Delaunay

    verts = np.asarray(vertices, dtype=float)
    rng = np.random.default_rng(seed)
    tri = Delaunay(verts)
    lo, hi = verts.min(0), verts.max(0)
    out = []
    while len(out) < n:
        batch = rng.uniform(lo, hi, size=(max(4 * n, 1024), 3))
        out.extend(batch[tri.find_simplex(batch) >= 0])
    return np.asarray(out[:n], dtype=float)


def evaluate_3d_field(A, vf, point) -> np.ndarray:
    """Evaluate a 3-D affine vector field at ``point``."""
    gens = A.S.gens()
    subs = {gens[i]: float(point[i]) for i in range(3)}
    if len(gens) > 3:
        subs[gens[3]] = 1
    return np.array([float(comp.subs(subs)) for comp in vf[:3]], dtype=float)


def evaluate_3d_field_batch(A, vf, points) -> np.ndarray:
    """Evaluate a 3-D affine vector field at many points."""
    return np.array([evaluate_3d_field(A, vf, p) for p in np.atleast_2d(points)], dtype=float)


def boundary_samples_3d(vertices, per_face: int = 200, seed: Optional[int] = None):
    """Sample points on each triangular boundary face with outward normals."""
    from scipy.spatial import ConvexHull

    verts = np.asarray(vertices, dtype=float)
    hull = ConvexHull(verts)
    rng = np.random.default_rng(seed)
    all_p, all_n, all_f = [], [], []
    for fi, (simp, eq) in enumerate(zip(hull.simplices, hull.equations)):
        normal = eq[:3] / np.linalg.norm(eq[:3])
        v0, v1, v2 = verts[simp]
        r = rng.random((per_face, 2))
        over = r.sum(1) > 1
        r[over] = 1 - r[over]
        pts = (
            np.outer(1 - r[:, 0] - r[:, 1], v0)
            + np.outer(r[:, 0], v1)
            + np.outer(r[:, 1], v2)
        )
        all_p.append(pts)
        all_n.append(np.tile(normal, (per_face, 1)))
        all_f.append(np.full(per_face, fi))
    return np.vstack(all_p), np.vstack(all_n), np.concatenate(all_f)


def unconstrained_basis(A, degree: int):
    """Return all polynomial planar vector fields up to ``degree``."""
    from sage.modules.free_module_element import vector

    v = A.S.gens()[:-1]
    zero = A.S.zero()
    basis = []
    for d in range(degree + 1):
        for a in range(d + 1):
            p = (v[0] ** a) * (v[1] ** (d - a))
            basis.append(vector([p, zero]))
            basis.append(vector([zero, p]))
    return basis


def unconstrained_basis_3d(A, degree: int):
    """Return all polynomial 3-D vector fields up to ``degree``."""
    from sage.modules.free_module_element import vector

    v = A.S.gens()[:-1]
    zero = A.S.zero()
    basis = []
    for d in range(degree + 1):
        for a in range(d + 1):
            for b in range(d - a + 1):
                c = d - a - b
                p = v[0] ** a * v[1] ** b * v[2] ** c
                basis.append(vector([p, zero, zero]))
                basis.append(vector([zero, p, zero]))
                basis.append(vector([zero, zero, p]))
    return basis


def pv_polyhedron_mesh(vertices):
    """Create a PyVista triangular surface mesh for a convex 3-D polyhedron."""
    from scipy.spatial import ConvexHull
    import pyvista as pv

    verts = np.asarray(vertices, dtype=float)
    hull = ConvexHull(verts)
    faces = []
    for simplex in hull.simplices:
        faces.extend([3, *simplex])
    return pv.PolyData(verts, faces=faces)


def add_arrows(plotter, origins, vectors, color="viridis", scale=1.0, scalar_name="speed", opacity=0.7, clim=None):
    """Add a PyVista arrow-glyph layer."""
    import pyvista as pv

    origins = np.asarray(origins, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    speed = np.linalg.norm(vectors, axis=1)
    cloud = pv.PolyData(origins)
    cloud["vectors"] = vectors * scale
    cloud[scalar_name] = speed
    arrows = cloud.glyph(orient="vectors", scale="vectors", factor=1.0)
    if isinstance(color, str) and color in ["viridis", "plasma", "coolwarm", "RdBu_r"]:
        plotter.add_mesh(arrows, scalars=scalar_name, cmap=color, opacity=opacity, clim=clim, show_scalar_bar=False)
    else:
        plotter.add_mesh(arrows, color=color, opacity=opacity, show_scalar_bar=False)


def add_observation_arrows(plotter, obs_dict, scale: float = 1.0) -> None:
    """Overlay observation arrows in red and singular observations as red points."""
    import pyvista as pv

    pts = np.array(list(obs_dict.keys()), dtype=float)
    vecs = np.array(list(obs_dict.values()), dtype=float)
    speed_sq = (vecs**2).sum(axis=1)

    nonzero = speed_sq > 1e-10
    if nonzero.any():
        add_arrows(plotter, pts[nonzero], vecs[nonzero], color="red", scale=scale, opacity=1.0)

    singular = ~nonzero
    if singular.any():
        sing_cloud = pv.PolyData(pts[singular])
        plotter.add_mesh(sing_cloud, color="red", point_size=100 * scale, render_points_as_spheres=True, opacity=0.5)
