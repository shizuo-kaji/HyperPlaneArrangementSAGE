"""
Sample: Compare polynomial approximations of different degrees.

Generate a high-degree (k=10) polynomial vector field from the logarithmic module,
then approximate it using lower degrees (k=4, 6, 8) to study approximation quality.

Run (after activating your Sage environment, e.g. ``conda activate sage``)::

    sage -python examples/polynomial_degree_comparison.sage
"""

try:
    from hyperplane_arrangements import (
        HyperplaneArrangement,
        graded_component,
        fit_vf,
        fit_vorticity,
    )
except ImportError as exc:
    raise SystemExit(
        "Install the package first via `sage -pip install -e .`"
    ) from exc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import griddata

# Define an irregular hexagon
hexagon = [
    (0.25, 0.15),
    (0.75, 0.10),
    (0.95, 0.55),
    (0.70, 0.90),
    (0.30, 0.85),
    (0.10, 0.40),
]

# Build the arrangement
A = HyperplaneArrangement(vertices=hexagon)
G = A.minimal_generators[1:]  # drop Euler

# Sample points inside the hexagon - define function first
def sample_random_points_in_polygon(vertices, n_points, seed=None, min_dist_from_boundary=0.03,
                                    min_dist_between_points=0.0):
    """Sample random points inside a polygon using rejection sampling.

    Parameters:
    -----------
    vertices : array-like
        Vertices of the polygon
    n_points : int
        Number of points to sample
    seed : int, optional
        Random seed for reproducibility
    min_dist_from_boundary : float
        Minimum distance from polygon edges (default: 0.05)
    min_dist_between_points : float
        Minimum distance between sampled points (default: 0.0 = no constraint)
    """
    if seed is not None:
        np.random.seed(seed)

    vertices = np.array(vertices)
    min_x, min_y = vertices.min(axis=0)
    max_x, max_y = vertices.max(axis=0)

    def distance_to_segment(p, v1, v2):
        """Compute distance from point p to line segment v1-v2."""
        # Vector from v1 to v2
        seg = v2 - v1
        # Vector from v1 to p
        v1p = p - v1

        # Project v1p onto seg
        seg_len_sq = np.dot(seg, seg)
        if seg_len_sq == 0:
            return np.linalg.norm(v1p)

        t = np.dot(v1p, seg) / seg_len_sq
        t = np.clip(t, 0, 1)  # Clamp to segment

        # Closest point on segment
        closest = v1 + t * seg
        return np.linalg.norm(p - closest)

    def min_distance_to_boundary(p, vertices):
        """Compute minimum distance from point p to any edge of the polygon."""
        n = len(vertices)
        min_dist = float('inf')
        for i in range(n):
            v1 = vertices[i]
            v2 = vertices[(i + 1) % n]
            dist = distance_to_segment(p, v1, v2)
            min_dist = min(min_dist, dist)
        return min_dist

    def min_distance_to_points(p, points):
        """Compute minimum distance from point p to existing points."""
        if len(points) == 0:
            return float('inf')
        dists = [np.linalg.norm(p - np.array(pt)) for pt in points]
        return min(dists)

    points = []
    max_attempts = n_points * 2000  # Increase max attempts for spacing constraint
    attempts = 0

    while len(points) < n_points and attempts < max_attempts:
        attempts += 1

        # Generate random point in bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        p = np.array([x, y])

        # Check if point is inside polygon using winding number algorithm
        inside = False
        n = len(vertices)
        j = n - 1
        for i in range(n):
            if ((vertices[i, 1] > y) != (vertices[j, 1] > y)) and \
               (x < (vertices[j, 0] - vertices[i, 0]) * (y - vertices[i, 1]) /
                (vertices[j, 1] - vertices[i, 1]) + vertices[i, 0]):
                inside = not inside
            j = i

        if inside:
            # Check distance to boundary
            dist_boundary = min_distance_to_boundary(p, vertices)
            if dist_boundary < min_dist_from_boundary:
                continue

            # Check distance to other points
            if min_dist_between_points > 0:
                dist_points = min_distance_to_points(p, points)
                if dist_points < min_dist_between_points:
                    continue

            points.append([x, y])

    if len(points) < n_points:
        print(f"Warning: Only generated {len(points)}/{n_points} points with constraints")
        print(f"  min_dist_from_boundary={min_dist_from_boundary}, min_dist_between_points={min_dist_between_points}")

    return np.array(points)

# Generate a high-degree polynomial vector field as ground truth
# First, create a flow field with vortices at specific points
k_true = 10
print(f"Generating ground truth polynomial field with degree k = {k_true}")
print("Creating vortex-based flow...")

# Define vortex centers (well inside the hexagon)
vortex_centers = [
    (0.35, 0.30),
#    (0.65, 0.25),
#    (0.80, 0.50),
#    (0.70, 0.50),
#    (0.40, 0.75),
#    (0.25, 0.55),
#    (0.50, 0.45),
#    (0.55, 0.60),
#    (0.45, 0.35),
#    (0.60, 0.40),
]

# Random vortex strengths
np.random.seed(42)
vortex_strengths = np.random.randn(len(vortex_centers)) * 2.0

print(f"Created {len(vortex_centers)} vortices with strengths: {vortex_strengths[:5]}...")

# Generate synthetic flow with vortices and fit with high-degree polynomials
# Sample points to capture the vortex flow
n_fitting_pts = 100
fitting_pts = sample_random_points_in_polygon(hexagon, n_fitting_pts, seed=456,
                                              min_dist_from_boundary=0.04,
                                              min_dist_between_points=0.05)

# Compute vortex-based velocity at each point
def vortex_velocity(pt, centers, strengths):
    """Compute velocity from point vortices."""
    vel = np.zeros(2)
    for center, strength in zip(centers, strengths):
        dx = pt[0] - center[0]
        dy = pt[1] - center[1]
        r2 = dx**2 + dy**2
        if r2 > 1e-6:  # Avoid singularity
            # Vortex velocity: v = (strength / 2π) * (-dy/r², dx/r²)
            factor = strength / (2 * np.pi * r2)
            vel[0] += -dy * factor
            vel[1] += dx * factor
    return vel

# Generate vortex flow observations
Obs_vortex = {}
for pt in fitting_pts:
    vel = vortex_velocity(pt, vortex_centers, vortex_strengths)
    Obs_vortex[tuple(pt)] = tuple(vel)

# Now fit the vortex flow with high-degree polynomials
mod_gens_true = graded_component(G, k_true)
print(f"Fitting vortex flow with k={k_true} polynomial basis...")

from hyperplane_arrangements.arrangement import fit_vf
vf_true, residual_true = fit_vf(A, Obs_vortex, mod_gens_true, verbose=False)
print(f"Polynomial approximation residual: {residual_true:.4e}")

print(f"\nGround truth field has {len(vf_true)} components")

# Sample measurement points (avoiding boundary to prevent singularities)
n_samples = 50
min_boundary_dist = 0.03  # Stay at least 0.03 units away from edges
min_point_spacing = 0.06  # Minimum distance between sample points
sample_pts = sample_random_points_in_polygon(hexagon, n_samples, seed=123,
                                             min_dist_from_boundary=min_boundary_dist,
                                             min_dist_between_points=min_point_spacing)
print(f"\nSampled {len(sample_pts)} measurement points")
print(f"  Min boundary distance: {min_boundary_dist}, Min point spacing: {min_point_spacing}")

# Evaluate ground truth field at sample points
def eval_vf(pt, vf, A):
    """Evaluate a vector field at a point."""
    subs_dict = {A.S.gens()[0]: pt[0], A.S.gens()[1]: pt[1]}
    if len(A.S.gens()) > 2:
        subs_dict[A.S.gens()[2]] = 1
    return np.array([float(comp.subs(subs_dict)) for comp in vf[:2]], dtype=float)

def eval_vorticity(pt, vf, A):
    """Evaluate vorticity: omega_z = dv_y/dx - dv_x/dy"""
    subs_dict = {A.S.gens()[0]: pt[0], A.S.gens()[1]: pt[1]}
    if len(A.S.gens()) > 2:
        subs_dict[A.S.gens()[2]] = 1

    x, y = A.S.gens()[0], A.S.gens()[1]
    dvy_dx = vf[1].derivative(x).subs(subs_dict)
    dvx_dy = vf[0].derivative(y).subs(subs_dict)

    return float(dvy_dx - dvx_dy)

# Generate observations from ground truth
Obs_vel = {}
Obs_vort = {}
n_valid = 0
for pt in sample_pts:
    vel = eval_vf(pt, vf_true, A)
    vort = eval_vorticity(pt, vf_true, A)

    # Check for NaN/inf values
    if not (np.all(np.isfinite(vel)) and np.isfinite(vort)):
        continue

    # Add small noise
    vel_noisy = vel + np.random.normal(0, 0.01 * np.linalg.norm(vel), size=2)
    vort_noisy = vort + np.random.normal(0, 0.01 * abs(vort))

    Obs_vel[tuple(pt)] = tuple(vel_noisy)
    Obs_vort[tuple(pt)] = vort_noisy
    n_valid += 1

print(f"Generated {len(Obs_vel)} velocity observations ({n_valid}/{len(sample_pts)} valid)")
print(f"Generated {len(Obs_vort)} vorticity observations")

# Update sample_pts to only include valid points
sample_pts = np.array(list(Obs_vel.keys()))
print(f"Using {len(sample_pts)} valid sample points")

# Fit using different degrees
degrees = [4,7,10]
#degrees = [4,5,6,7,8,9,10]
fits_vel = {}
fits_vort = {}
residuals_vel = {}
residuals_vort = {}

for k in degrees:
    print(f"\n--- Fitting with degree k = {k} ---")
    mod_gens = graded_component(G, k)

    # Fit from velocity
    print(f"Fitting from velocity data...")
    vf_fit_vel, res_vel = fit_vf(A, Obs_vel, mod_gens, verbose=False)
    fits_vel[k] = vf_fit_vel
    residuals_vel[k] = res_vel
    print(f"  Velocity fit residual: {res_vel:.4e}")

    # Fit from vorticity
    print(f"Fitting from vorticity data...")
    vf_fit_vort, res_vort = fit_vorticity(A, Obs_vort, mod_gens, verbose=False)
    fits_vort[k] = vf_fit_vort
    residuals_vort[k] = res_vort
    print(f"  Vorticity fit residual: {res_vort:.4e}")

# Evaluate all fields at sample points
true_vecs = np.array([eval_vf(pt, vf_true, A) for pt in sample_pts])
true_vort = np.array([eval_vorticity(pt, vf_true, A) for pt in sample_pts])

fit_vecs_vel = {k: np.array([eval_vf(pt, fits_vel[k], A) for pt in sample_pts]) for k in degrees}
fit_vecs_vort = {k: np.array([eval_vf(pt, fits_vort[k], A) for pt in sample_pts]) for k in degrees}

fit_vort_vel = {k: np.array([eval_vorticity(pt, fits_vel[k], A) for pt in sample_pts]) for k in degrees}
fit_vort_vort = {k: np.array([eval_vorticity(pt, fits_vort[k], A) for pt in sample_pts]) for k in degrees}

# Compute errors
print("\n--- Error Metrics ---")
print("Velocity fits:")
for k in degrees:
    vel_err = np.sqrt(np.mean((true_vecs - fit_vecs_vel[k])**2))
    vort_err = np.sqrt(np.mean((true_vort - fit_vort_vel[k])**2))
    print(f"  k={k}: Velocity RMSE = {vel_err:.6e}, Vorticity RMSE = {vort_err:.6e}")

print("\nVorticity fits:")
for k in degrees:
    vel_err = np.sqrt(np.mean((true_vecs - fit_vecs_vort[k])**2))
    vort_err = np.sqrt(np.mean((true_vort - fit_vort_vort[k])**2))
    print(f"  k={k}: Velocity RMSE = {vel_err:.6e}, Vorticity RMSE = {vort_err:.6e}")

# Visualization
def plot_quiver(ax, points, vectors, title, color):
    pts = np.array(points)
    vecs = np.array(vectors)
    ax.quiver(pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1], color=color, angles='xy', scale=20, width=0.005)
    ax.scatter(pts[:, 0], pts[:, 1], c='black', s=10, alpha=0.5, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(title)

def plot_vorticity_contour(ax, points, vorticity, title, cmap='RdBu_r'):
    pts = np.array(points)
    vort = np.array(vorticity)

    vmax = max(abs(vort.min()), abs(vort.max()))
    if vmax == 0:
        vmax = 1
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Create interpolated background heatmap/contour
    # Generate a grid covering the point cloud
    x_min, y_min = pts.min(axis=0) - 0.02
    x_max, y_max = pts.max(axis=0) + 0.02
    grid_x, grid_y = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )

    # Interpolate vorticity values to the grid
    grid_vort = griddata(pts, vort, (grid_x, grid_y), method='cubic', fill_value=np.nan)

    # Plot interpolated contour as background
    contour = ax.contourf(grid_x, grid_y, grid_vort, levels=20, cmap=cmap, norm=norm, alpha=0.6)

    # Plot scatter points on top to show measurement locations
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=vort, cmap=cmap, norm=norm, s=60,
                        edgecolors='black', linewidths=0.5, zorder=10)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Vorticity')
    return scatter

def plot_streamlines(ax, vf, A, vertices, title, color='blue', density=1.5):
    """Plot streamlines of a vector field."""
    # Create a grid for streamline computation
    vertices = np.array(vertices)
    min_x, min_y = vertices.min(axis=0) - 0.05
    max_x, max_y = vertices.max(axis=0) + 0.05

    # Create grid
    nx, ny = 50, 50
    x = np.linspace(min_x, max_x, nx)
    y = np.linspace(min_y, max_y, ny)
    X, Y = np.meshgrid(x, y)

    # Evaluate vector field on grid
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    for i in range(nx):
        for j in range(ny):
            pt = np.array([X[j, i], Y[j, i]])
            # Check if point is inside polygon
            inside = point_in_polygon(pt, vertices)
            if inside:
                try:
                    vel = eval_vf(pt, vf, A)
                    if np.all(np.isfinite(vel)):
                        U[j, i] = vel[0]
                        V[j, i] = vel[1]
                    else:
                        U[j, i] = 0
                        V[j, i] = 0
                except:
                    U[j, i] = 0
                    V[j, i] = 0
            else:
                U[j, i] = np.nan
                V[j, i] = np.nan

    # Plot streamlines
    speed = np.sqrt(U**2 + V**2)
    ax.streamplot(X, Y, U, V, color=color, density=density, linewidth=1,
                  arrowsize=1.2, arrowstyle='->')

    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

def point_in_polygon(pt, vertices):
    """Check if point is inside polygon using winding number."""
    x, y = pt
    n = len(vertices)
    inside = False
    j = n - 1
    for i in range(n):
        if ((vertices[i, 1] > y) != (vertices[j, 1] > y)) and \
           (x < (vertices[j, 0] - vertices[i, 0]) * (y - vertices[i, 1]) /
            (vertices[j, 1] - vertices[i, 1]) + vertices[i, 0]):
            inside = not inside
        j = i
    return inside

# Create comparison figure with 3 rows
fig, axes = plt.subplots(3, len(degrees) + 1, figsize=(24, 15))

# Row 1: Vector fields
plot_quiver(axes[0, 0], sample_pts, true_vecs, f"True field", 'tab:blue')
for idx, k in enumerate(degrees):
    vel_rmse = np.sqrt(np.mean((true_vecs - fit_vecs_vel[k])**2))
    plot_quiver(axes[0, idx+1], sample_pts, fit_vecs_vel[k],
                f"Velocity fit (k={k})\nRMSE={vel_rmse:.4e}", 'tab:green')

# Row 2: Vorticity
plot_vorticity_contour(axes[1, 0], sample_pts, true_vort, f"True vorticity")
for idx, k in enumerate(degrees):
    vort_rmse = np.sqrt(np.mean((true_vort - fit_vort_vort[k])**2))
    plot_vorticity_contour(axes[1, idx+1], sample_pts, fit_vort_vort[k],
                          f"Vorticity fit (k={k})\nRMSE={vort_rmse:.4e}")

# Row 3: Streamlines
print("\nGenerating streamline plots...")
plot_streamlines(axes[2, 0], vf_true, A, hexagon, f"True streamlines", color='blue')
for idx, k in enumerate(degrees):
    vel_rmse = np.sqrt(np.mean((true_vecs - fit_vecs_vel[k])**2))
    plot_streamlines(axes[2, idx+1], fits_vel[k], A, hexagon,
                    f"Streamlines (k={k})\nRMSE={vel_rmse:.4e}", color='green')

# Draw polygon boundary
poly = np.vstack([hexagon, hexagon[0]])
for ax_row in axes:
    for ax in ax_row:
        ax.plot(poly[:, 0], poly[:, 1], 'k-', linewidth=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

#fig.suptitle(f'Polynomial Degree Comparison: Approximating k={k_true} vortex-based field\n'
#             f'Rows: Vector field (quiver) | Vorticity | Streamlines | {len(sample_pts)} sample points',
#             fontsize=13, y=0.99)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('polynomial_degree_comparison.png', dpi=300)
#plt.show()
plt.close()

# Create error convergence plot
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

# Velocity errors
vel_errors_from_vel = [np.sqrt(np.mean((true_vecs - fit_vecs_vel[k])**2)) for k in degrees]
vel_errors_from_vort = [np.sqrt(np.mean((true_vecs - fit_vecs_vort[k])**2)) for k in degrees]

axes2[0].semilogy(degrees, vel_errors_from_vel, 'o-', label='Fit from velocity', linewidth=2, markersize=8)
axes2[0].semilogy(degrees, vel_errors_from_vort, 's-', label='Fit from vorticity', linewidth=2, markersize=8)
axes2[0].set_xticks(degrees)
axes2[0].set_xlabel('Polynomial degree k', fontsize=12)
axes2[0].set_ylabel('Velocity RMSE', fontsize=12)
axes2[0].set_title('Velocity Approximation Error', fontsize=13)
axes2[0].grid(True, alpha=0.3)
axes2[0].legend()

# Vorticity errors
vort_errors_from_vel = [np.sqrt(np.mean((true_vort - fit_vort_vel[k])**2)) for k in degrees]
vort_errors_from_vort = [np.sqrt(np.mean((true_vort - fit_vort_vort[k])**2)) for k in degrees]

axes2[1].semilogy(degrees, vort_errors_from_vel, 'o-', label='Fit from velocity', linewidth=2, markersize=8)
axes2[1].semilogy(degrees, vort_errors_from_vort, 's-', label='Fit from vorticity', linewidth=2, markersize=8)
axes2[1].set_xticks(degrees)
axes2[1].set_xlabel('Polynomial degree k', fontsize=12)
axes2[1].set_ylabel('Vorticity RMSE', fontsize=12)
axes2[1].set_title('Vorticity Approximation Error', fontsize=13)
axes2[1].grid(True, alpha=0.3)
axes2[1].legend()

fig2.suptitle(f'Error Convergence: Approximating k={k_true} polynomial field', fontsize=13)
plt.tight_layout()
plt.savefig('polynomial_degree_convergence.png', dpi=300)
#plt.show()
plt.close()
