"""
Sample: fit a polynomial vector field to synthetic vorticity data on a convex polygon.

Run (after activating your Sage environment, e.g. ``conda activate sage``)::

    sage -python examples/vorticity_fit_demo.sage
"""

try:
    from hyperplane_arrangements import (
        HyperplaneArrangement,
        graded_component,
        fit_vorticity,
        fit_vf,
        ConvexPolygonFlow,
    )
except ImportError as exc:
    raise SystemExit(
        "Install the package first via `sage -pip install -e .`"
    ) from exc

# Define an irregular hexagon for the synthetic flow
hexagon = [
    (0.25, 0.15),
    (0.75, 0.10),
    (0.95, 0.55),
    (0.70, 0.90),
    (0.30, 0.85),
    (0.10, 0.40),
]

# Generate a tangential flow and sample both vorticity and velocity data
flow = ConvexPolygonFlow(hexagon, n_vortices=5, projection_length=0.08, seed=7)

# Sample vorticity values
Obs_vort = flow.sample_vorticity(30, noise=5e-3)
print(f"Collected {len(Obs_vort)} vorticity samples")

# Sample velocity vectors at the same points for comparison
Obs_vel = {}
for pt, _ in Obs_vort.items():
    vel = flow.velocity(pt)
    # Add small noise to velocity measurements
    import numpy as np
    vel_noisy = vel + np.random.normal(0, 5e-3, size=2)
    Obs_vel[pt] = tuple(vel_noisy)
print(f"Collected {len(Obs_vel)} velocity samples")

# Build the arrangement directly from vertices (auto-homogenised)
A = HyperplaneArrangement(vertices=hexagon)
G = A.minimal_generators[1:]  # drop Euler

k = max(2, G[0][0].degree())
print(f"\nUsing graded component degree k = {k}")
mod_gens = graded_component(G, k)

# Fit using vorticity measurements
print("\n--- Fitting from vorticity data ---")
vf_vort_fit, residual_vort = fit_vorticity(A, Obs_vort, mod_gens, verbose=True)
print(f"Vorticity fit residual = {residual_vort:.4e}")

# Fit using direct velocity measurements
print("\n--- Fitting from velocity data ---")
vf_vel_fit, residual_vel = fit_vf(A, Obs_vel, mod_gens, verbose=True)
print(f"Velocity fit residual = {residual_vel:.4e}")

# Inspect both fitted fields at a random interior point
pt = tuple(flow.sample_random_points(1)[0])
print(f"\nRandom test point: {pt}")
true_vel = flow.velocity(pt)

subs_dict = {A.S.gens()[0]: pt[0], A.S.gens()[1]: pt[1]}
if len(A.S.gens()) > 2:
    subs_dict[A.S.gens()[2]] = 1

# Only use first 2 components for 2D vector field
fitted_vel_vort = [comp.subs(subs_dict) for comp in vf_vort_fit[:2]]
fitted_vel_vel = [comp.subs(subs_dict) for comp in vf_vel_fit[:2]]

print("True velocity:", true_vel)
print("Fitted velocity (from vorticity):", [float(val) for val in fitted_vel_vort])
print("Fitted velocity (from velocity):", [float(val) for val in fitted_vel_vel])

# -----------------------------
# Visual comparison
# -----------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


def plot_quiver(ax, points, vectors, title, color):
    pts = np.array(points)
    vecs = np.array(vectors)
    # Increase arrow width and scale for better visibility with fewer points
    ax.quiver(pts[:, 0], pts[:, 1], vecs[:, 0], vecs[:, 1], color=color, angles='xy', scale=20, width=0.005)
    # Add markers at the sampled points
    ax.scatter(pts[:, 0], pts[:, 1], c='black', s=10, alpha=0.5, zorder=5)
    ax.set_aspect('equal')
    ax.set_title(title)


def eval_fitted_vf(pt, vf):
    """Evaluate a fitted vector field at a point."""
    subs_dict = {A.S.gens()[0]: pt[0], A.S.gens()[1]: pt[1]}
    if len(A.S.gens()) > 2:
        subs_dict[A.S.gens()[2]] = 1
    # Only return the first 2 components (x, y) for 2D vector field
    return np.array([float(comp.subs(subs_dict)) for comp in vf[:2]], dtype=float)


def eval_fitted_vorticity(pt, vf):
    """Evaluate fitted vorticity: omega_z = dv_y/dx - dv_x/dy"""
    subs_dict = {A.S.gens()[0]: pt[0], A.S.gens()[1]: pt[1]}
    if len(A.S.gens()) > 2:
        subs_dict[A.S.gens()[2]] = 1

    # Compute partial derivatives
    x, y = A.S.gens()[0], A.S.gens()[1]
    dvy_dx = vf[1].derivative(x).subs(subs_dict)
    dvx_dy = vf[0].derivative(y).subs(subs_dict)

    return float(dvy_dx - dvx_dy)


def plot_vorticity_contour(ax, points, vorticity, title, cmap='RdBu_r'):
    pts = np.array(points)
    vort = np.array(vorticity)

    # Create a scatter plot with color representing vorticity
    vmax = max(abs(vort.min()), abs(vort.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Increase marker size for better visibility with fewer points
    scatter = ax.scatter(pts[:, 0], pts[:, 1], c=vort, cmap=cmap, norm=norm, s=60, edgecolors='black', linewidths=0.5)
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Vorticity')
    return scatter


# Use the sampled points for evaluation (where measurements were taken)
sample_pts = np.array(list(Obs_vort.keys()))
true_vecs = np.array([flow.velocity(tuple(pt)) for pt in sample_pts])

# Evaluate both fitted vector fields at sampled points
fit_vecs_vort = np.array([eval_fitted_vf(pt, vf_vort_fit) for pt in sample_pts])
fit_vecs_vel = np.array([eval_fitted_vf(pt, vf_vel_fit) for pt in sample_pts])

# Compute differences
diff_vecs_vort = true_vecs - fit_vecs_vort
diff_vecs_vel = true_vecs - fit_vecs_vel

# Compute vorticity at sampled points
true_vort = np.array([flow.vorticity(tuple(pt)) for pt in sample_pts])
fit_vort_from_vort = np.array([eval_fitted_vorticity(pt, vf_vort_fit) for pt in sample_pts])
fit_vort_from_vel = np.array([eval_fitted_vorticity(pt, vf_vel_fit) for pt in sample_pts])

# Compute vorticity differences
diff_vort_from_vort = true_vort - fit_vort_from_vort
diff_vort_from_vel = true_vort - fit_vort_from_vel

# Print error metrics
print("\n--- Error Metrics ---")
print(f"Velocity RMSE (vorticity fit): {np.sqrt(np.mean(diff_vecs_vort**2)):.4e}")
print(f"Velocity RMSE (velocity fit):  {np.sqrt(np.mean(diff_vecs_vel**2)):.4e}")
print(f"Vorticity RMSE (vorticity fit): {np.sqrt(np.mean(diff_vort_from_vort**2)):.4e}")
print(f"Vorticity RMSE (velocity fit):  {np.sqrt(np.mean(diff_vort_from_vel**2)):.4e}")

# Create figure with 2 rows x 4 columns
fig, axes = plt.subplots(2, 4, figsize=(20, 10))

# Row 1: Vector fields (at sampled points)
plot_quiver(axes[0, 0], sample_pts, true_vecs, "True velocity field", 'tab:blue')
plot_quiver(axes[0, 1], sample_pts, fit_vecs_vort, "Fit from vorticity", 'tab:green')
plot_quiver(axes[0, 2], sample_pts, fit_vecs_vel, "Fit from velocity", 'tab:orange')
plot_quiver(axes[0, 3], sample_pts, diff_vecs_vel, "Velocity fit error", 'tab:red')

# Row 2: Vorticity (at sampled points)
plot_vorticity_contour(axes[1, 0], sample_pts, true_vort, "True vorticity")
plot_vorticity_contour(axes[1, 1], sample_pts, fit_vort_from_vort, "From vorticity fit")
plot_vorticity_contour(axes[1, 2], sample_pts, fit_vort_from_vel, "From velocity fit")
plot_vorticity_contour(axes[1, 3], sample_pts, diff_vort_from_vort, "Vorticity fit error", cmap='seismic')

# Draw polygon boundary on all subplots
poly = np.vstack([hexagon, hexagon[0]])
for ax_row in axes:
    for ax in ax_row:
        ax.plot(poly[:, 0], poly[:, 1], 'k-', linewidth=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')

# Add error metrics as text
vel_rmse_vort = np.sqrt(np.mean(diff_vecs_vort**2))
vel_rmse_vel = np.sqrt(np.mean(diff_vecs_vel**2))
vort_rmse_vort = np.sqrt(np.mean(diff_vort_from_vort**2))
vort_rmse_vel = np.sqrt(np.mean(diff_vort_from_vel**2))

fig.suptitle(f'Comparison: Vorticity Fit vs Direct Velocity Fit (at {len(sample_pts)} sampled points)\n'
             f'Velocity RMSE: {vel_rmse_vort:.2f} (vorticity) vs {vel_rmse_vel:.2f} (velocity) | '
             f'Vorticity RMSE: {vort_rmse_vort:.2f} (vorticity) vs {vort_rmse_vel:.2f} (velocity)',
             fontsize=12, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
