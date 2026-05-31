import pytest
from sage.all import matrix, QQ, GF, vector, PolynomialRing, identity_matrix
from hyperplane_arrangements.arrangement import (
    HyperplaneArrangement,
)
from hyperplane_arrangements.utils import coordinate_vectors

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

def test_constructive_closure_and_s_invariant():
    A = HyperplaneArrangement(matrix(QQ, [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, -1, 0],
    ]))

    closure, closure_indices = A.constructive_closure([0, 1, 2], return_indices=True)
    assert closure_indices == (0, 1, 2, 3)
    assert closure.num_planes == 4

    B, indices = A.minimal_constructive_subset(return_indices=True)
    assert indices == (0, 1, 2)
    assert B.num_planes == 3
    assert A.constructively_generates(indices)
    assert A.s_invariant() == 3
    assert A.s_invariant() == 3

def test_constructive_closure_is_trivial_in_rank_two():
    A = HyperplaneArrangement(matrix(QQ, [[1, 0], [0, 1], [1, 1]]))
    B, indices = A.minimal_constructive_subset(return_indices=True)
    assert indices == (0, 1, 2)
    assert B.num_planes == 3
    assert A.constructive_closure_indices([0, 1]) == (0, 1)
    assert A.s_invariant() == 3


def test_constructive_closure_Lk_parameter():
    # R^4: boolean e0..e3 plus v = e0 + e1 (idx 4) and w = e2 + e3 (idx 5)
    A = HyperplaneArrangement(matrix(QQ, [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 0, 1, 1],
    ]))
    # L_2: the pair (e0, e1) spans a rank-2 flat contained in v -> v is pulled in
    assert A.constructive_closure_indices([0, 1], k=2) == (0, 1, 4)
    # L_3: two hyperplanes cannot form a rank-3 flat, so the closure is trivial
    assert A.constructive_closure_indices([0, 1], k=3) == (0, 1)
    assert A.constructive_closure_indices([0, 1, 2, 3], k=2) == (0, 1, 2, 3, 4, 5)
    assert A.constructively_generates([0, 1, 2, 3], k=2)
    # default k == n - 1 == 3 (historical behaviour preserved)
    assert A.s_invariant() == A.s_invariant(3)


def test_s_invariant_depends_on_k():
    # rank-2 arrangement embedded in R^4: every normal lies in span(e0, e1)
    A = HyperplaneArrangement(matrix(QQ, [
        [1, 0, 0, 0], [0, 1, 0, 0], [1, 1, 0, 0], [1, -1, 0, 0],
    ]))
    # L_2: any two independent hyperplanes span the whole rank-2 flat -> all four
    assert A.constructive_closure_indices([0, 1], k=2) == (0, 1, 2, 3)
    assert A.s_invariant(2) == 2
    # L_3: there is no rank-3 flat at all, so nothing is ever pulled in
    assert A.s_invariant(3) == 4
    assert A.s_invariant() == 4  # default k = n - 1 = 3


def test_intersection_lattice_basic_central_arrangement():
    A = HyperplaneArrangement(matrix(QQ, [[1, 0], [0, 1], [1, 1]]))
    L = A.intersection_lattice()

    expected_elements = {
        tuple(),
        (0,),
        (1,),
        (2,),
        (0, 1, 2),
    }

    assert set(L) == expected_elements
    assert L.bottom() == tuple()
    assert L.top() == (0, 1, 2)
    assert L.is_lequal((0,), (0, 1, 2))
    assert not L.is_lequal((0, 1, 2), (0,))


def test_intersection_lattice_shim_function():
    A = HyperplaneArrangement(matrix(QQ, [[1, 0], [0, 1], [1, 1]]))
    direct = A.intersection_lattice()
    shimmed = A.intersection_lattice()

    assert set(direct) == set(shimmed)

def test_dimension_formula():
    mat = matrix(QQ, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    A = HyperplaneArrangement(mat)
    assert A.vf_dimension(2) == 3
    assert A.vf_dimension(3) == 8

def test_ntf2_dimension_4_arrangements():
    from hyperplane_arrangements.utils import coordinate_vectors

    mat1 = [[ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
            [ 0,  0,  1,  0],
            [-1,  1,  0,  0],
            [-1,  0,  1,  0],
            [ 0, -1,  1,  0],
            [ 0,  0,  0,  1],
            [ 0,  1, -1,  1],
            [-1,  1, -1,  1],
            [ 0,  0, -1,  1]]
    A1 = HyperplaneArrangement(matrix(QQ, mat1))
    assert A1.num_planes == 10
    assert A1.degrees() == [1, 3, 3, 3]

    mat2 = [[1,0,0,0],[1,1,0,0],[1,0,1,0],[1,0,0,1],
            [1,-1,0,0],[1,0,-1,0],[1,0,0,-1],
            [0,1,-1,0],[0,0,1,-1],[0,1,0,1]]
    A2 = HyperplaneArrangement(matrix(QQ, mat2))
    assert A2.num_planes == 10
    assert A2.degrees() == [1, 3, 3, 3]

    B2 = A2.deletion([1,2])
    assert B2.num_planes == 8
    assert B2.degrees() == [1, 3, 3, 3, 3, 3, 3]

    mat3 = coordinate_vectors(4) + [[1 ,-1,0, 0],[1,0 ,-1, 0],[1,0,0,-1],
                           [0 ,1, -1,0],[0,1 ,0, -1],[0,0,1,-1],
                           [0,1 ,-1, 1],[1,-1,1,-1]]
    A3 = HyperplaneArrangement(matrix(QQ, mat3))
    assert A3.num_planes == 12
    assert A3.degrees() == [1, 3, 4, 4]

    B3_1 = A3.deletion([0,1])
    assert B3_1.num_planes == 10
    assert B3_1.degrees() == [1, 3, 3, 4, 4]

    B3_2 = A3.deletion([0,7])
    assert B3_2.num_planes == 10
    assert B3_2.degrees() == [1, 3, 4, 4, 4, 5]

    B3_3 = A3.deletion([1,9])
    assert B3_3.num_planes == 10
    assert B3_3.degrees() == [1, 3, 4, 4, 4, 4]


# --- Notebook cell 3: Fano plane over QQ and GF(2) ---

def test_fano_plane_QQ():
    seed = [1,1,0,1,0,0,0]
    mat = [seed.copy()]
    for i in range(len(seed)-1):
        l = seed.pop(0)
        seed.append(l)
        mat.append(seed.copy())
    A = HyperplaneArrangement(mat, base_field=QQ)
    assert A.degrees() == [1, 1, 1, 1, 1, 1, 1]


def test_fano_plane_GF2():
    seed = [1,1,0,1,0,0,0]
    mat = [seed.copy()]
    for i in range(len(seed)-1):
        l = seed.pop(0)
        seed.append(l)
        mat.append(seed.copy())
    A = HyperplaneArrangement(mat, base_field=GF(2))
    assert A.degrees() == [0, 0, 0, 1, 2, 3, 3, 3]


# --- Notebook cell 5: free arrangement dim=3, |A|=3 ---

def test_free_coord_arrangement_3():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1]]))
    assert A.is_free
    assert A.degrees() == [1, 1, 1]
    MG = A.minimal_generators()
    assert len(MG) == 3
    for g in MG:
        assert A.is_in_DA(g)


# --- Notebook cell 7: arrangement from defining polynomial ---

def test_arrangement_from_Q_3d():
    R = PolynomialRing(QQ, 'x', 3)
    x, y, z = R.gens()
    A = HyperplaneArrangement(Q=x*y*z)
    assert A.num_planes == 3
    assert A.n == 3
    assert A.is_free
    assert A.degrees() == [1, 1, 1]


# --- Notebook cell 10: generators are in D(A) ---

def test_generators_are_in_DA():
    R = PolynomialRing(QQ, 'x', 3)
    x, y, z = R.gens()
    A = HyperplaneArrangement(Q=x*y*z*(x+y+z))
    assert A.num_planes == 4
    for g in A.minimal_generators():
        assert A.is_in_DA(g)


# --- Notebook cell 6, 11: free resolution ---

def test_free_resolution_free():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1]]))
    res = A.free_resolution()
    assert res._length == 1  # free => length 1


def test_free_resolution_spog():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    res = A.free_resolution()
    assert res._length == 2  # SPOG => length 2


# --- Notebook cell 14, 58: SPOG and level_coeff ---

def test_is_spog():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    spog = A.is_spog()
    assert spog  # truthy
    assert sorted(spog) == [1, 2, 2, 2]


def test_level_coeff():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    (c, i), RHS = A.level_coeff()
    MG = A.minimal_generators()
    # the relation: c*MG[i] == sum of c_j*MG[j]
    lhs = c * MG[i].v
    rhs = sum(cj * MG[j].v for cj, j in RHS)
    assert lhs == rhs


# --- Notebook cell 13: second differential gives relations (zero check) ---

def test_free_resolution_differential_relation():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    res = A.free_resolution()
    # columns of differential(1) are the generators; their product with differential(2) is zero
    d1 = res.differential(1).matrix()
    d2 = res.differential(2).matrix()
    assert d1 * d2 == 0


# --- Notebook cell 19: deletion with multiple indices ---

def test_deletion_multiple_indices():
    mat = [[ 1,  0,  0,  0],
           [ 0,  1,  0,  0],
           [ 0,  0,  1,  0],
           [-1,  1,  0,  0],
           [-1,  0,  1,  0],
           [ 0, -1,  1,  0],
           [ 0,  0,  0,  1],
           [ 0,  1, -1,  1],
           [-1,  1, -1,  1],
           [ 0,  0, -1,  1]]
    A = HyperplaneArrangement(matrix(QQ, mat))
    B = A.deletion([2, 4, 9])
    assert B.num_planes == 7
    assert B.degrees() == [1, 2, 2, 2]


# --- Notebook cells 25-27: Orlik-Terao Ex4.36 free ⊂ non-free ⊂ free ---

def test_orlik_terao_ex436():
    A = HyperplaneArrangement(matrix(QQ, coordinate_vectors(3)+[[1,1,-1],[1,1,0]]))
    assert A.is_free
    assert A.degrees() == [1, 2, 2]

    B = A.deletion([4])
    assert not B.is_free
    assert B.degrees() == [1, 2, 2, 2]

    C = B.deletion([3])
    assert C.is_free
    assert C.degrees() == [1, 1, 1]


# --- Notebook cell 29: restriction ---

def test_restriction():
    A = HyperplaneArrangement(matrix(QQ, [
        [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1],[1,1,1,-1]
    ]))
    B = A.restriction(4)
    assert B.num_planes > 0
    # restriction by index and by vector should agree
    H = A.mat[4]
    C = A.restriction(H)
    assert B.num_planes == C.num_planes


# --- Notebook cell 23: euler_complement splits off Euler ---

def test_euler_complement():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    MG = A.minimal_generators()
    # after complement, the last coordinate should vanish for non-Euler gens
    for g in MG[1:]:
        gc = A.euler_complement(g.v, 0)
        assert gc[0] == 0


# --- Notebook cell 34: parametrised family, free iff b == -t ---

def test_parametrised_family_free():
    b = 2
    t2 = 3
    # t = -b => free
    A = HyperplaneArrangement(matrix(QQ,
        coordinate_vectors(3)+[[1,1,0],[b,0,1],[0,-b,1],[0,t2,1]]))
    assert A.is_free
    assert A.degrees() == [1, 3, 3]

    # t != -b => not free
    A2 = HyperplaneArrangement(matrix(QQ,
        coordinate_vectors(3)+[[1,1,0],[b,0,1],[0,1,1],[0,t2,1]]))
    assert not A2.is_free


# --- Notebook cell 36: degree sequence depends on base field ---

def test_degs_depend_on_base_field():
    mat = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,1,1],[1,0,-1],[0,1,1]]
    A = HyperplaneArrangement(mat, base_field=QQ)
    B = HyperplaneArrangement(mat, base_field=GF(2))
    assert A.degrees() == [1, 3, 3]
    assert B.degrees() == [1, 2, 4]


# --- Notebook cell 48: generic arrangement degree pattern ---

def test_generic_arrangement_degree_pattern():
    n = 3
    mat = matrix(QQ, coordinate_vectors(n)+[[1,1,1],[2,3,1],[3,4,1]])
    A = HyperplaneArrangement(mat)
    # p=6, generic in dim 3: each step adds p-n generators
    assert A.num_planes == 6
    assert A.degrees() == [1, 4, 4, 4, 4, 4]


def test_scaled_minor_ideal_for_more_than_ell_plus_one_generators():
    n = 3
    A = HyperplaneArrangement(matrix(QQ, coordinate_vectors(n)+[[1,1,1],[2,3,1],[3,4,1]]))
    G = list(A.minimal_generators())

    coeffs = A.saito_coefficients(G)
    assert isinstance(coeffs, dict)
    assert len(coeffs) == 20  # C(6, 3)

    assert A.scaled_minor_ideal(G) == A.S.ideal(A.determinant_ideal())
    assert A.scaled_minor_ideal_height(G) >= 0

    with pytest.raises(ValueError, match="ell \\+ 1 generators"):
        A.check_generalized_saito(generators=G)


def test_saito_coefficients_keep_signed_list_for_ell_plus_one_generators():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1],[1,1,2]]))
    coeffs = A.saito_coefficients()

    assert isinstance(coeffs, list)
    assert len(coeffs) == A.n + 1
    assert A.scaled_minor_ideal() == A.S.ideal([c for c in coeffs if c != 0])


# --- Notebook cell 59: addition makes arrangement free ---

def test_addition_free():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    assert not A.is_free
    B = A.addition([1, 0, 1])
    assert B.is_free
    assert B.degrees() == [1, 2, 2]
    C = A.addition([1, 1, 0])
    assert C.is_free
    assert C.degrees() == [1, 2, 2]


# --- Notebook cell 60: relations (syz) ---

def test_relations():
    A = HyperplaneArrangement(matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]]))
    rels = A.relations()
    assert len(rels) > 0


# --- Notebook cell 31: multi-arrangement (restriction with multiplicity) ---

def test_multi_arrangement_restriction():
    A = HyperplaneArrangement(matrix(QQ, [
        [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1],[1,1,1,-1]
    ]))
    B = A.restriction(3)
    assert B.multiplicity is not None
    MG = B.compute_multi_minimal_generators()
    assert MG.degrees() == [2, 2, 2, 2]


# --- Notebook cell 32: deletion then restriction ---

def test_deletion_then_restriction():
    A = HyperplaneArrangement(matrix(QQ, [
        [1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[1,1,1,1],[1,1,1,-1]
    ]))
    B = A.deletion([0]).restriction(1)
    assert B.multiplicity == [1, 1, 1, 1]
    MG = B.compute_multi_minimal_generators()
    assert MG.degrees() == [1, 1, 2]


# --- Notebook cell 61: SPOG with higher degrees ---

def test_spog_higher_degree():
    A = HyperplaneArrangement(
        identity_matrix(QQ, 3).stack(matrix([
            [1,1,0],[0,1,-1],[1,1,1],[1,1,2],[1,1,-2],[1,-1,0]
        ]))
    )
    spog = A.is_spog()
    assert spog
    assert sorted(spog) == [1, 4, 5, 5]
    (c, i), RHS = A.level_coeff()
    MG = A.minimal_generators()
    lhs = c * MG[i].v
    rhs = sum(cj * MG[j].v for cj, j in RHS)
    assert lhs == rhs


# --- Notebook cell 64-65: arrangement with no free addition, SPOG counter-example ---

def test_not_free_arrangement():
    A = HyperplaneArrangement(matrix(QQ,
        coordinate_vectors(3)+[[1,1,0],[1,3,1],[0,3,1],[0,2,1],[0,1,1],[0,1,-1],[0,1,2]]))
    assert A.degrees() == [1, 3, 6]
    assert A.is_free


def test_spog_level_coeff_in_arrangement():
    """SPOG whose level coeff is already in the arrangement (counter-example to NT-free minus)."""
    A = HyperplaneArrangement(matrix(QQ,
        coordinate_vectors(3)+[[1,1,0],[1,3,1],[0,3,1],[0,2,1],[0,1,1],[0,1,-1],[0,1,2]]))
    B = A.deletion([1, 5])
    spog = B.is_spog()
    assert spog
    assert sorted(spog) == [1, 3, 5, 6]


# --- Notebook cell 78-79: free resolution length (projective dimension) ---

def test_free_resolution_pd_dim5():
    A = HyperplaneArrangement(matrix(QQ, coordinate_vectors(5)+[[1,1,1,1,1]]))
    res = A.free_resolution()
    assert res._length == 4  # pd = 4


# --- Notebook cell 70: Ziegler pair (same lattice, same degree seq) ---

def test_ziegler_pair_same_degs():
    M = [[0,1,0],[2,2,1],[3,1,1],[8,-1,4],[9,3,-1]]
    A = HyperplaneArrangement(matrix(QQ, M+[[9,-2,3],[11,2,1],[5,5,-2]]))
    B = HyperplaneArrangement(matrix(QQ, M+[[21,-4,7],[19,4,1],[10,10,-5]]))
    assert A.degrees() == [1, 5, 5, 5, 5]
    assert B.degrees() == [1, 5, 5, 5, 5]


# --- basis_da and alternative algorithms agree ---

def test_basis_da_agrees_with_minimal_generators():
    mat = matrix(QQ, [[1,0,0],[0,1,0],[0,0,1],[1,1,1]])
    A = HyperplaneArrangement(mat)
    expected = A.degrees()
    MG2 = A.compute_basis_syzygy(verbose=False)
    assert MG2.degrees() == expected
    MG3 = A.compute_basis_linear(verbose=False)
    assert MG3.degrees() == expected


# --- Notebook cell 50-51: near-pencil arrangements max deg = p-n+1 ---

def test_near_pencil_max_deg():
    n = 5
    for p in [6, 7, 8]:
        mat = matrix(QQ, coordinate_vectors(n)+[[t]+[1]*(n-1) for t in range(1, p-n+1)])
        A = HyperplaneArrangement(mat)
        assert A.num_planes == p
        assert max(A.degrees()) <= p - n + 1


def test_multi_arrangement_all_counts_one_dim3():
    mat = matrix(QQ, coordinate_vectors(3) + [[1, 1, 1], [1, -1, 0]])
    A = HyperplaneArrangement(mat)
    A_multi = HyperplaneArrangement(mat, multiplicity=[1] * A.num_planes)
    
    expected_degrees = A.degrees()
    multi_gens = A_multi.compute_multi_minimal_generators()
    assert multi_gens.degrees() == expected_degrees


def test_multi_arrangement_all_counts_one_dim4():
    mat = matrix(QQ, coordinate_vectors(4) + [[1, 1, 1, 1], [1, -1, 1, -1]])
    A = HyperplaneArrangement(mat)
    A_multi = HyperplaneArrangement(mat, multiplicity=[1] * A.num_planes)
    
    expected_degrees = A.degrees()
    multi_gens = A_multi.compute_multi_minimal_generators()
    assert multi_gens.degrees() == expected_degrees
