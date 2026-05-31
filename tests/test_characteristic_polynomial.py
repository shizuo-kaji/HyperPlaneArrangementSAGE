import pytest
from sage.all import matrix, QQ
from hyperplane_arrangements.arrangement import HyperplaneArrangement

def test_characteristic_polynomial():
    # Braid arrangement A_3 in R^3: t^3 - 4t^2 + 6t - 3
    # Hyperplanes: x_1=0, x_2=0, x_3=0, x_1+x_2+x_3=0 (generic actually, so boolean + 1 generic)
    mat = matrix(QQ, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    A = HyperplaneArrangement(mat)
    poly = A.characteristic_polynomial()
    
    assert poly.degree() == 3
    t = poly.parent().gen()
    expected = t**3 - 4*t**2 + 6*t - 3
    assert poly == expected

    # Empty arrangement in R^3
    mat_empty = matrix(QQ, 0, 3)
    A_empty = HyperplaneArrangement(mat_empty)
    poly_empty = A_empty.characteristic_polynomial()
    expected_empty = t**3
    assert poly_empty == expected_empty
    
    # 1 plane in R^2: t^2 - t
    mat_1 = matrix(QQ, [[1, 0]])
    A_1 = HyperplaneArrangement(mat_1)
    poly_1 = A_1.characteristic_polynomial()
    expected_1 = t**2 - t
    assert poly_1 == expected_1
