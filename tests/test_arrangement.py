import pytest
from sage.all import matrix, QQ, vector
from hyperplane_arrangements.arrangement import HyperplaneArrangement

def test_arrangement_from_matrix():
    mat = matrix(QQ, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    A = HyperplaneArrangement(mat)
    assert A.n == 3
    assert A.num_planes == 4
    assert not A.is_free

def test_arrangement_euler():
    mat = matrix(QQ, [[1, 0], [0, 1], [1, 1]])
    A = HyperplaneArrangement(mat)
    e = A.euler()
    assert len(e) == 2
    assert A.is_in_DA(e)

def test_arrangement_jacob_I():
    mat = matrix(QQ, [[1, 0], [0, 1]])
    A = HyperplaneArrangement(mat)
    jac = A.jacob_I()
    assert len(jac.gens()) == 2

def test_arrangement_from_vertices():
    vertices = [(0, 0), (1, 0), (0, 1)]
    A = HyperplaneArrangement(vertices=vertices, base_field=QQ)
    assert A.num_planes == 4
    assert A.n == 3

def test_arrangement_from_Q():
    from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
    S = PolynomialRing(QQ, 'x', 2)
    x = S.gens()
    Q = x[0] * x[1] * (x[0] + x[1])
    A = HyperplaneArrangement(Q=Q)
    assert A.num_planes == 3
    assert A.n == 2

def test_mutually_exclusive_args():
    with pytest.raises(ValueError, match="provide exactly one of"):
        mat = matrix(QQ, [[1, 0], [0, 1]])
        HyperplaneArrangement(mat=mat, vertices=[(0, 0), (1, 0), (0, 1)])

def test_linear_forms():
    mat = matrix(QQ, [[1, 0], [0, 1], [1, 1]])
    A = HyperplaneArrangement(mat)
    forms = A.linear_forms()
    assert len(forms) == 3
    assert forms[0] == A.v[0]
    assert forms[1] == A.v[1]
    assert forms[2] == A.v[0] + A.v[1]

def test_dimension_formula():
    mat = matrix(QQ, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    A = HyperplaneArrangement(mat)
    assert A.vf_dimension(2) == 3
    assert A.vf_dimension(3) == 8
