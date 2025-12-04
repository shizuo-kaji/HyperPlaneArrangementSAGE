"""
Practical example: Fit a logarithmic vector field to fluid simulation data.

Requirements:
- First run the Python simulation to generate CSV:
    python3 flowsim_numpy.py
  This creates: fluid_sim_final_state.csv (x, y, vx, vy, vorticity, divergence)

This Sage script will:
1) Recreate the irregular hexagon used in the simulation
2) Build a projective arrangement from its 6 affine edges (3 variables: x, y, z)
3) Compute minimal generators of D(A)
4) Build a graded basis up to chosen degree k and fit to observed (vx, vy)
5) Optionally add divergence-free or rotation-free soft constraints in the fit
6) Report residual and save a plot of the fitted field over the polygon

Run in Sage:
    sage -python - <<'PY'
or load in a Sage notebook and run.
"""

try:
    from hyperplane_arrangements import *  # packaged installation
except ImportError:
    load("../logarithmic_vector_fields.py")

import csv
import numpy as np
import matplotlib.pyplot as plt

print("--- Practical example: Fit to fluid simulation over hexagon ---")

# Fit options (can override via env or argv)
#   ENFORCE = 'none' | 'div' | 'rot'
#     - 'div': divergence-free soft constraint (∇·u ≈ 0)
#     - 'rot': rotation-free soft constraint (curl_z(u) ≈ 0 in 2D)
#   LAMBDA: penalty weight for constraints (larger = stronger enforcement)
#   DEGREE: graded component degree k to use (int)
#   MIN_DIST: min distance to any boundary line for sampling (float)
#   MAX_POINTS: cap number of interior points sampled (int)
import os, sys, random
ENFORCE = os.environ.get('ENFORCE', 'none')
LAMBDA = float(os.environ.get('LAMBDA', '1e-2'))
DEGREE = int(os.environ.get('DEGREE', '0'))  # 0 => auto from generators
MIN_DIST = float(os.environ.get('MIN_DIST', '1e-3'))
MAX_POINTS = int(os.environ.get('MAX_POINTS', '3000'))

# Simple argv parsing: key=value
for arg in sys.argv[1:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        if k == 'ENFORCE':
            ENFORCE = v
        elif k == 'LAMBDA':
            LAMBDA = float(v)
        elif k == 'DEGREE':
            DEGREE = int(v)
        elif k == 'MIN_DIST':
            MIN_DIST = float(v)
        elif k == 'MAX_POINTS':
            MAX_POINTS = int(v)
        
print(f"Fit config -> ENFORCE={ENFORCE}, LAMBDA={LAMBDA}, MIN_DIST={MIN_DIST}, MAX_POINTS={MAX_POINTS}")

# 1) Recreate the irregular hexagon (must match flowsim_numpy.py)
X_MAX = 1.0
Y_MAX = 1.0
center_x, center_y = X_MAX/2, Y_MAX/2
base_radius = min(X_MAX, Y_MAX) * 0.4
radii_factors = [1.0, 0.9, 1.1, 1.0, 0.95, 1.05]
verts = []
for i in range(6):
    angle = pi/3 * i + pi/6
    r = base_radius * radii_factors[i]
    verts.append((float(center_x + r*cos(angle)), float(center_y + r*sin(angle))))

print("Hexagon vertices:")
for v in verts:
    print("  ", v)

# 2) Build projective arrangement from edges: each affine line through two points
#    In homogeneous coords, the line through p1=(x1,y1,1), p2=(x2,y2,1) is p1 x p2 = (a,b,c)
def hom_line(p1, p2):
    (x1, y1), (x2, y2) = p1, p2
    # cross product (x1,y1,1) x (x2,y2,1)
    a = y1 - y2
    b = x2 - x1
    c = x1*y2 - x2*y1
    # Quantise to modest rationals to avoid huge coefficients downstream
    def qrat(val, nd=6):
        import fractions
        f = fractions.Fraction(val).limit_denominator(10**nd)
        return QQ(f.numerator) / QQ(f.denominator)
    return (qrat(a), qrat(b), qrat(c))

lines_abc = []
for i in range(6):
    p1 = verts[i]
    p2 = verts[(i+1) % 6]
    lines_abc.append(hom_line(p1, p2))

mat = matrix(QQ, lines_abc)  # 6 x 3, central in P^2
A = HyperPlaneArr(mat)

print(f"Ambient dimension n (projective variables x,y,z): {A.n}")
print(f"Number of lines: {A.num_planes}")
print(f"Polynomial ring: {A.S}")
print(f"Defining polynomial Q (product of 6 lines): degree {A.Q.degree()}")

# 3) Minimal generators (exclude Euler later for fitting)
MG = A.minimal_generators
print(f"Minimal generator degrees: {A.degs}")

# 4) Read observations (vx, vy) at (x,y) from CSV
csv_path = "../fluid_sim_final_state.csv"
Obs = {}
num_loaded = 0
with open(csv_path, "r") as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        x = float(row["x"]) ; y = float(row["y"]) ; vx = float(row["vx"]) ; vy = float(row["vy"])
        # ignore points very close to boundary to avoid ill-conditioning (tunable)
        # heuristic: distance to each line a*x + b*y + c = 0
        d_min = min(abs(a*x + b*y + c)/sqrt(a*a + b*b) for (a,b,c) in lines_abc)
        if d_min >= MIN_DIST:
            Obs[(x, y)] = vector([vx, vy])
            num_loaded += 1

# Optional random subsampling for experiments
if len(Obs) > MAX_POINTS:
    keys = list(Obs.keys())
    random.shuffle(keys)
    keep = set(keys[:MAX_POINTS])
    Obs = {k: Obs[k] for k in keep}
    print(f"Subsampled to {len(Obs)} points (MAX_POINTS)")

print(f"Loaded {num_loaded} interior observations from {csv_path}")

# 5) Choose degree and fit: use graded components up to k
non_euler_gens = MG[1:]  # drop Euler
if len(non_euler_gens) == 0:
    raise ValueError("No non-Euler generators to fit.")

k = DEGREE if DEGREE > 0 else max(2, non_euler_gens[0][0].degree())  # default >=2
print(f"Fitting with graded component degree k = {k}")
mod_gens = graded_component(non_euler_gens, k)
print(f"Basis size for fitting: {len(mod_gens)}")

if ENFORCE == 'none':
    # Unconstrained fitting via library helper
    u_fit, res = fit_vf(A, Obs, mod_gens, verbose=True)
    print(f"Residual (sum of squares) = {res}")
    out_png = "practical_hexagon_fit.png"
else:
    # Constrained fit (soft penalty) on dehomogenised basis
    # Build dehomogenised, S-independent basis Gd
    Gd = vector_basis(dehomogenise(A, mod_gens))
    print(f"Basis dimension (dehomogenised): {len(Gd)}")

    # Build data matrices
    v2 = A.S.gens()[:-1]  # (x,y)
    import numpy as _np

    # A_data rows: for each point (vx row, vy row)
    rows = []
    rhs = []
    for (x, y), vel in Obs.items():
        subs_q = {v2[0]: RR(x), v2[1]: RR(y)}
        # Evaluate basis at (x,y)
        bu = [] ; bv = []
        for b in Gd:
            bu.append(float(RR(b[0].subs(subs_q))))
            bv.append(float(RR(b[1].subs(subs_q))))
        rows.append(_np.array(bu, dtype=float)) ; rhs.append(float(vel[0]))
        rows.append(_np.array(bv, dtype=float)) ; rhs.append(float(vel[1]))

    A_data = _np.vstack(rows)
    b_data = _np.array(rhs)

    # Constraint matrix C for divergence or rotation sampled at same points
    # div(b) = du/dx + dv/dy, rot(b) = dv/dx - du/dy
    Cv_rows = []
    zero_rhs = []
    x_sym, y_sym = v2
    for (x, y) in Obs.keys():
        subs_q = {x_sym: RR(x), y_sym: RR(y)}
        if ENFORCE == 'div':
            row = []
            for b in Gd:
                val = diff(b[0], x_sym).subs(subs_q) + diff(b[1], y_sym).subs(subs_q)
                row.append(float(RR(val)))
            Cv_rows.append(_np.array(row, dtype=float))
            zero_rhs.append(0.0)
        elif ENFORCE == 'rot':
            row = []
            for b in Gd:
                val = diff(b[1], x_sym).subs(subs_q) - diff(b[0], y_sym).subs(subs_q)
                row.append(float(RR(val)))
            Cv_rows.append(_np.array(row, dtype=float))
            zero_rhs.append(0.0)
        else:
            raise ValueError(f"Unknown ENFORCE mode: {ENFORCE}")

    C = _np.vstack(Cv_rows)
    d = _np.array(zero_rhs)

    # Solve augmented LS: minimize ||A c - b||^2 + LAMBDA ||C c||^2
    A_aug = _np.vstack([A_data, (LAMBDA**0.5)*C])
    b_aug = _np.hstack([b_data, (LAMBDA**0.5)*d])
    c_hat, *_ = _np.linalg.lstsq(A_aug, b_aug, rcond=None)

    # Build fitted field as linear combo of basis Gd
    u_fit = (matrix(Gd).transpose()) * vector(c_hat)

    # Report residuals
    res_main = float(_np.sum((A_data.dot(c_hat) - b_data)**2))
    res_constr = float(_np.sum((C.dot(c_hat) - d)**2))
    print(f"Residual data = {res_main}, constraint = {res_constr} (lambda={LAMBDA})")

    out_png = f"practical_hexagon_fit_{ENFORCE}.png"
    # Also save/copy a canonical name for acceptance
    try:
        import shutil
        shutil.copyfile(out_png, "practical_hexagon_fit.png")
    except Exception:
        pass

# 6) Plot arrangement and fitted vector field (dehomogenised to z=1)
try:
    ax = A.plot_vfield(u_fit, quiver=True, nx=40, ny=40, xlim=(0, 1), ylim=(0, 1))
    plt.plot([v[0] for v in verts] + [verts[0][0]], [v[1] for v in verts] + [verts[0][1]], 'k-', lw=1.0)
    plt.savefig(out_png, dpi=200)
    print(f"Saved: {out_png}")
except Exception as e:
    print(f"Plotting skipped due to: {e}")

print("Done.")
