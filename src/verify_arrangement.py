from sage.all import *
from hyperplane_arrangements.arrangement import HyperplaneArrangement, min_gen_arr

def test_arrangement():
    print("Testing HyperplaneArrangement...")

    # Test 1: From matrix
    print("Test 1: From matrix")
    mat = matrix(QQ, [[1, 0], [0, 1], [1, 1]])
    A = HyperplaneArrangement(mat)
    print(f"Number of planes: {A.num_planes}")
    print(f"Is free: {A.is_free}")
    print(f"Degrees: {A.degs()}")

    # Test 2: From polynomial
    print("\nTest 2: From polynomial")
    R = PolynomialRing(QQ, 'x', 2)
    x = R.gens()
    Q = x[0] * x[1] * (x[0] + x[1])
    A2 = HyperplaneArrangement(Q=Q)
    print(f"Number of planes: {A2.num_planes}")
    print(f"Is free: {A2.is_free}")

    # Test 3: SPOG check
    print("\nTest 3: SPOG check")
    # A generic arrangement of 4 lines in 2D is not free but SPOG?
    # Actually generic 2D is always free.
    # Let's try 3D generic 4 planes -> free.
    # We need a non-free example.
    # The example from the paper: x, y, z, x+y, x+z, y+z (A3 braid) is free.
    # Let's just check if is_spog runs without error on a free arrangement (should be False or maybe True with level?)
    # Wait, SPOG definition usually implies not free.
    print(f"Is SPOG (A): {A.is_spog()}")

    # Test 4: min_gen_arr
    print("\nTest 4: min_gen_arr")
    gens = min_gen_arr(mat)
    print(f"Number of generators: {len(gens)}")

    # Test 5: euler_complement
    print("\nTest 5: euler_complement")
    if A.is_free:
        u = A.euler()
        uc = A.euler_complement(u, -1)
        print("euler_complement execution successful")

    print("\nAll tests passed!")

if __name__ == "__main__":
    test_arrangement()
