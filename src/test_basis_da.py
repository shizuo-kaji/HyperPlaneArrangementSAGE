from sage.all import *
from hyperplane_arrangements.arrangement import basis_da

# Test basis_da directly
mat = matrix(QQ, [[1, 0], [0, 1], [1, 1]])
print(f"Matrix:\n{mat}")
print(f"\nTesting basis_da for degree 1:")
basis1 = basis_da(mat, 1)
print(f"Number of basis elements: {len(basis1)}")
print(f"Basis: {basis1}")

print(f"\nTesting basis_da for degree 2:")
basis2 = basis_da(mat, 2)
print(f"Number of basis elements: {len(basis2)}")
print(f"Basis: {basis2}")
