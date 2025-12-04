"""
Conceptual example: A2-like arrangement in P^2 with four lines
  {x = 0, y = 0, z = 0, x + y + z = 0}.

What this script does
- Build the arrangement and compute minimal generators
- Compute a free resolution and show Betti tables
- Compute graded dimensions dim_Q D(A)_k for small k via generators
- Verify membership of generators in D(A)
- Plot a non-Euler generator vector field and save PNG

Run
  sage -python examples/conceptual_example.sage
  # or in a Sage notebook: load("examples/conceptual_example.sage")
"""

try:
    from hyperplane_arrangements import *  # packaged installation
except ImportError:
    load("../logarithmic_vector_fields.py")

# Define the central arrangement in QQ^3 by its linear forms' coefficient rows
# Lines: x=0, y=0, z=0, x+y+z=0 in homogeneous coords (x,y,z)
mat = matrix(QQ, [
    [1, 0, 0],  # x = 0
    [0, 1, 0],  # y = 0
    [0, 0, 1],  # z = 0
    [1, 1, 1],  # x + y + z = 0
])

A = HyperPlaneArr(mat)

print("--- Conceptual example: A2-like arrangement in P^2 ---")
print(f"Ambient dimension n: {A.n}")
print(f"Number of lines: {A.num_planes}")
print(f"Polynomial ring: {A.S}")
print(f"Defining polynomial Q: {A.Q}")

# Minimal generators and degrees
MG = A.minimal_generators
degs = A.degs
print(f"Minimal generator degrees: {degs}")
print(f"Is free? {A.is_free}")

# Free resolution and Betti numbers table (if available)
try:
    fr = A.free_resolution()
    # Show Betti numbers per homological degree
    print("Betti numbers (homological degree -> degree: multiplicity):")
    try:
        # Some Sage objects expose ._length; iterate 0.._length
        L = fr._length
    except Exception:
        # Fallback to 3 steps
        L = 3
    for i in range(L+1):
        try:
            print(f"  {i}: {fr.betti(i)}")
        except Exception:
            pass
except Exception as e:
    print(f"free_resolution not available: {e}")

# Compute graded component dimensions using generators
print("Graded dimensions dim_Q D(A)_k (computed from generators):")
for k in [1, 2, 3, 4]:
    try:
        comp = graded_component(MG, k)
        # Over QQ, number of S-independent elements equals dimension at degree k
        dimk = len(vector_basis(comp))
        print(f"  k={k}: {dimk}")
    except Exception as e:
        print(f"  k={k}: failed to compute ({e})")

# Verify that each minimal generator lies in D(A)
print("Membership checks in D(A):")
for i, g in enumerate(MG):
    try:
        ok = A.is_in_DA(g)
    except Exception as e:
        ok = f"error: {e}"
    print(f"  MG[{i}] in D(A)? {ok}")

# Choose a non-Euler generator and plot on dehomogenised chart z=1
non_euler = MG[1] if len(MG) > 1 else MG[0]
print(f"Non-Euler generator (symbolic): {non_euler}")

try:
    ax = A.plot_vfield(non_euler, quiver=True, nx=25, ny=25, xlim=(-2, 2), ylim=(-2, 2))
    import matplotlib.pyplot as plt
    plt.savefig("conceptual_A2_vector_field.png", dpi=200)
    print("Saved: conceptual_A2_vector_field.png")
except Exception as e:
    print(f"Plotting skipped due to: {e}")

print("Done.")
