import itertools
import numpy as np
from sage.rings.rational_field import QQ
from sage.rings.integer_ring import ZZ
from sage.matrix.constructor import matrix, zero_matrix
from sage.modules.free_module_element import zero_vector
from sage.combinat.composition import Compositions
from sage.matrix.args import MatrixArgs
from sage.libs.singular.function_factory import singular_function
from sage.structure.sequence import Sequence

minbase = lambda L: singular_function('minbase')(Sequence(L))

def coord_vec(n):
    r"""Return the list of coordinate vectors in ``QQ^n``."""
    return list(np.eye(n))

def is_distinct_planes(mat):
    r"""Check whether the rows of ``mat`` define distinct hyperplanes."""
    for plane in itertools.combinations(range(mat.nrows()), 2):
        if mat[plane, :].rank() < 2:
            return False
    return True

def remove_duplicate_planes(mat):
    r"""Remove duplicate or zero hyperplanes while tracking multiplicities."""
    indices = list(range(mat.nrows()))
    K = mat.base_ring()
    n = mat.ncols()
    zerovec = zero_vector(K, n)
    for plane in list(indices):
        if mat[plane] == zerovec:
            indices.remove(plane)
    multiplicity = {i: 1 for i in indices}
    for pair in itertools.combinations(indices, 2):
        if (pair[1] in indices) and (mat[pair, :].rank() < 2):
            indices.remove(pair[1])
            multiplicity[pair[0]] += 1
    return mat[indices, :], [multiplicity[i] for i in indices]

def create_generic_arrangement(n, k):
    r"""Return a random generic arrangement in ``QQ^n`` with ``k`` planes."""
    from sage.all import random_vector
    E = coord_vec(n) + [[1]*n]
    while len(E) < k:
        dependent = True
        while dependent:
            dependent = False
            rndvec = random_vector(ZZ, n)
            for J in itertools.combinations(E, n - 1):
                dp = (matrix(ZZ, J).stack(rndvec)).det()
                if dp == 0:
                    dependent = True
                    break
            if not dependent:
                E.append(rndvec)
    return E[:k]

def sk_expo(k, n):
    """
    Enumerate all exponents of monomials in S_k.
    """
    Sk_list = []
    ones = np.ones(n, dtype=int)
    for e in Compositions(k + n, length=n):
        Sk_list.append(tuple(np.array(e) - ones))
    return Sk_list

def exponent_to_polynomial(Ls, v):
    """
    Convert a list of exponents to a polynomial in indeterminants v[].
    """
    if not isinstance(Ls, list):
        Ls = [Ls]
    from sage.misc.misc_c import prod
    result = sum(prod(v[i]**L[i] for i in range(len(L))) for L in Ls)
    return result

def coef_map(k, S):
    """
    Compute matrices representing multiplication by x[i]: S_k to S_{k+1}.
    """
    v = S.gens()
    K = S.base_ring()
    n = len(v)

    Sk_list = sk_expo(k, n)
    Sk1_list = sk_expo(k + 1, n)
    Sk1_dic = {ex: i for i, ex in enumerate(Sk1_list)}

    Mi = [[] for i in range(n)]
    eye = np.eye(n)
    for i, expo in enumerate(Sk_list):
        for j in range(n):
            e = tuple(np.array(expo) + eye[j])
            Mi[j].append((i, Sk1_dic[e]))

    M = [MatrixArgs(K, len(Sk_list), len(Sk1_list),
                   entries={k: 1 for k in Mi[j]}).matrix()
         for j in range(n)]

    return M, [exponent_to_polynomial(expo, v) for expo in Sk1_list]

def _seq_to_matrix(seq):
    """Convert a Singular result (matrix or Sequence of vectors) to a Sage matrix."""
    if hasattr(seq, 'sage'):
        return seq.sage()
    from sage.matrix.constructor import matrix as mat_ctor
    return mat_ctor([list(row) for row in seq])


def module_intersection(modules):
    """Compute the intersection of multiple modules."""
    if not modules:
        raise ValueError("Empty list of modules")

    import sage.libs.singular.function_factory as singular
    result = modules[0].transpose()
    for mod in modules[1:]:
        result = singular.singular_function('intersect')(result, mod.transpose())

    return _seq_to_matrix(minbase(result)).transpose()
