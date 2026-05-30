"""Sage utilities for logarithmic vector fields on hyperplane arrangements."""
# ruff: noqa
# pylint: skip-file
import itertools
from copy import copy
from fractions import Fraction
from typing import List, Dict, Tuple, Union, Optional, Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.spatial import ConvexHull, QhullError

# Sage imports
from sage.all import kernel,Rational,singular,random_vector,factor,diff
from sage.structure.sage_object import SageObject
from sage.misc.cachefunc import cached_method
from sage.rings.integer_ring import ZZ
from sage.rings.integer import Integer
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.ideal import Ideal
from sage.matrix.constructor import matrix, zero_matrix, random_matrix
from sage.modules.free_module_element import vector, zero_vector
from sage.arith.functions import lcm
from sage.arith.misc import gcd
from sage.functions.other import binomial
from sage.functions.generalized import sgn
from sage.misc.latex import latex
from sage.combinat.composition import Compositions
from sage.libs.singular.function_factory import singular_function, lib as singular_lib
from sage.matrix.args import MatrixArgs
from sage.modules.free_module_element import FreeModuleElement_generic_dense as module_elem
from sage.structure.element import is_Vector, Vector, Element
from sage.structure.sequence import Sequence

from sage.misc.misc_c import prod

from .utils import (
    minbase,
    is_distinct_planes,
    remove_duplicate_planes,
    sk_expo,
    exponent_to_polynomial,
    coef_map,
    module_intersection,
    coordinate_vectors,
    create_generic_arrangement,
)
from .vector_field import VectorField, VectorFieldModule

singular_lib('presolve.lib')
syz = singular_function("syz")
ideal = Ideal

class HyperplaneArrangement(SageObject):
    r"""
    The main class for the logarithmic derivation module of a central arrangement.
    ...
    """

    def __init__(self, mat=None, *, vertices=None, multiplicity=None, Q=None,
                 base_field=QQ, vertex_max_denominator=10**6):
        provided = sum(x is not None for x in (mat, Q, vertices))
        if provided != 1:
            raise ValueError('provide exactly one of ``mat``, ``vertices``, or defining polynomial ``Q``.')

        if vertices is not None:
            mat = self._matrix_from_vertices(vertices, base_field=base_field,
                                             max_denominator=vertex_max_denominator)

        if mat is not None and Q is None:
            # arrangement from matrix
            if hasattr(mat, 'nrows') and hasattr(mat, 'base_ring'):
                self.mat = mat
            else:
                self.mat = matrix(base_field, mat)
            if not is_distinct_planes(self.mat):
                raise ValueError('the given arrangement contains duplicated planes!')
            self.K = self.mat.base_ring()
            self.n = self.mat.ncols()
            self.S = PolynomialRing(self.K, 'x', self.n)
            self.v = self.S.gens()
            self.Q = prod([sum(u[j]*self.v[j] for j in range(self.n)) for u in self.mat.rows()]) # defining polynomial
        elif mat is None and Q is not None:
            # arrangement from the defining polynomial
            self.Q = Q
            self.S = Q.parent()
            self.v = self.S.gens()
            self.K = self.S.base_ring()
            self.n = len(self.v)
            try:
                QF = Q.factor()
            except Exception:
                # If factorization fails, treat Q as a single factor
                QF = [(Q, 1)]
            B = []
            for linear_factor, exponent in QF:
                if linear_factor.parent() != self.S:
                    continue
                # Skip constant factors
                if not hasattr(linear_factor, 'degree'):
                    continue
                if linear_factor.degree() == 0:
                    continue
                # Extract coefficients for linear factors
                B.append([linear_factor.coefficient(self.v[i]) for i in range(self.n)])
            self.mat = matrix(self.K, B)
        else:
            raise ValueError('provide either matrix ``mat`` or defining polynomial ``Q``.')

        if not self.K.is_field():
            raise ValueError("The base ring must be a field.")

        ## parameters
        self.multiplicity = multiplicity
        self.num_planes = self.mat.nrows() # number of hyperplanes in the arrangement

    @classmethod
    def cone_of_lines(cls, normals, lines_by_dir, base_field=QQ):
        r"""Construct the central cone ``coning(A)`` of a 2D affine line
        arrangement as a rank-3 :class:`HyperplaneArrangement`.

        The affine arrangement is given in the pure-Python representation used
        by :mod:`hyperplane_arrangements.minimal_region`: ``normals`` is a list
        of integer direction vectors ``(a, b)`` and ``lines_by_dir[i]`` lists
        the offsets ``c`` of the lines ``a*x + b*y = c`` of direction
        ``normals[i]``.  Each such line cones to the plane ``a*x + b*y - c*z = 0``
        (note the sign: the offset sits on the right-hand side), and the line at
        infinity ``z = 0`` is appended.

        For a free affine arrangement the cone is free with exponents
        ``(1, d_1, d_2)`` -- recoverable via :meth:`degrees` -- which is the
        Yoshinaga setting ``chamber(A) = (1 + d_1)(1 + d_2)``.
        """
        normals = list(normals)
        lines_by_dir = list(lines_by_dir)
        if len(normals) != len(lines_by_dir):
            raise ValueError('normals and lines_by_dir must have the same length')
        rows = []
        for (a, b), offsets in zip(normals, lines_by_dir):
            for c in offsets:
                cc = Fraction(c)
                rows.append([
                    base_field(int(a)),
                    base_field(int(b)),
                    base_field(-cc.numerator) / base_field(cc.denominator),
                ])
        rows.append([base_field(0), base_field(0), base_field(1)])  # H_infty: z = 0
        return cls(matrix(base_field, rows))

    @classmethod
    def yoshinaga_multi_bound(cls, normals, lines_by_dir, base_field=QQ):
        r"""Compute the Yoshinaga multi-arrangement bound ``(1 + d_1)(1 + d_2)``.

        Builds the 2D central multi-arrangement from ``normals`` with multiplicities
        given by the lengths of offsets in ``lines_by_dir``, computes its multi-minimal
        generators, and returns the bound from their degrees ``d_1, d_2``.
        """
        normals = list(normals)
        lines_by_dir = list(lines_by_dir)
        if len(normals) != len(lines_by_dir):
            raise ValueError('normals and lines_by_dir must have the same length')
        rows = [[base_field(int(a)), base_field(int(b))] for a, b in normals]
        counts = [len(offsets) for offsets in lines_by_dir]
        arr = cls(matrix(base_field, rows), multiplicity=counts)
        d1, d2 = arr.compute_multi_minimal_generators().degrees()
        return (1 + int(d1)) * (1 + int(d2))

    @cached_method
    def euler(self):
        return VectorField(vector(self.v), self.S)

    @cached_method
    def jacob_I(self):
        return ideal(*[self.Q.derivative(self.v[j]) for j in range(self.n)])

    @cached_method
    def syz(self):
        M = self.minimal_generators()[0].v.parent()
        return M.submodule(self.jacob_I().syzygy_module())

    @cached_method
    def minimal_generators(self):
        # compute minimal generators of D(A) by D(A) \cong <Euler> \oplus Syz(J_Q)
        gens = [self.euler()] + [self.euler_complement(u) for u in minbase(self.jacob_I().syzygy_module())]
        return VectorFieldModule(gens)

    @cached_method
    def vf_dimension(self, k):
        fr = self.syz().graded_free_resolution()
        d = self.n - 1
        dim = 0
        for i in range(1, fr._length + 1):
            term_sum = 0
            for deg, mult in fr.betti(i).items():
                if k - deg >= 0:
                    term_sum += mult * binomial(k - deg + d, d)
            dim -= ((-1)**i) * term_sum
        return ZZ(dim)

    @cached_method
    def degrees(self):
        return self.minimal_generators().degrees()

    @property
    def is_free(self):
        return len(self.degrees()) == self.n

    @cached_method
    def linear_forms(self):
        return [sum(self.mat[i, j]*self.v[j] for j in range(self.n)) for i in range(self.num_planes)]

    @staticmethod
    def _matrix_from_vertices(vertices, *, base_field=QQ, max_denominator=10**6):
        if ConvexHull is None:
            raise ImportError('scipy.spatial.ConvexHull is required for vertex initialisation')
        try:
            vertex_list = [list(v) for v in vertices]
        except TypeError as exc:
            raise ValueError('``vertices`` must be an iterable of coordinate iterables') from exc
        if not vertex_list:
            raise ValueError('``vertices`` cannot be empty')
        dim = len(vertex_list[0])
        if dim < 1:
            raise ValueError('vertices must have at least one coordinate')
        for idx, vertex in enumerate(vertex_list):
            if len(vertex) != dim:
                raise ValueError(f'vertex {idx} has dimension {len(vertex)}, expected {dim}')

        try:
            points = np.array([[float(coord) for coord in vertex]
                               for vertex in vertex_list], dtype=float)
        except Exception as exc:
            raise ValueError('vertices must contain numeric entries') from exc

        if points.shape[0] <= dim:
            raise ValueError('need at least dim+1 vertices to determine a convex hull')

        try:
            hull = ConvexHull(points)
        except QhullError as exc:
            raise ValueError('convex hull construction failed; check for degeneracy or repeated points') from exc

        if hull.equations.size == 0:
            raise ValueError('convex hull has no supporting planes')

        rows = []
        seen = set()
        for eq in hull.equations:
            coeffs = [HyperplaneArrangement._coerce_vertex_scalar(val, base_field, max_denominator)
                      for val in eq[:-1]]
            coeffs.append(HyperplaneArrangement._coerce_vertex_scalar(eq[-1], base_field, max_denominator))
            coeffs = HyperplaneArrangement._normalise_plane_row(coeffs, base_field)
            key = tuple(coeffs)
            if key not in seen:
                seen.add(key)
                rows.append(coeffs)

        if not rows:
            raise ValueError('no hyperplanes constructed from vertices')

        n_homog = len(rows[0])
        h_inf = [base_field(0)] * (n_homog - 1) + [base_field(1)]
        rows.insert(0, h_inf)

        return matrix(base_field, rows)

    @staticmethod
    def _coerce_vertex_scalar(value, base_field, max_denominator):
        if hasattr(value, 'parent'):
            try:
                return base_field(value)
            except (TypeError, ValueError):
                pass
        if base_field == QQ and isinstance(value, (float, np.floating)):
            frac = Fraction(float(value)).limit_denominator(max_denominator)
            return QQ(frac.numerator) / QQ(frac.denominator)
        return base_field(value)

    @staticmethod
    def _normalise_plane_row(coeffs, base_field):
        if base_field != QQ:
            return coeffs

        denominators = [c.denominator() for c in coeffs if c]
        scale = 1
        for d in denominators:
            scale = lcm(scale, d)
        scaled = [c * scale for c in coeffs]
        numerators = [c.numerator() for c in scaled]

        gcd_val = 0
        for num in numerators:
            if num != 0:
                gcd_val = num if gcd_val == 0 else gcd(gcd_val, num)
        if gcd_val == 0:
            gcd_val = 1

        numerators = [num // gcd_val for num in numerators]

        sign = 1
        for num in numerators:
            if num != 0:
                sign = 1 if num > 0 else -1
                break

        return [base_field(sign * num) for num in numerators]

    def is_in_DA(self, gv):
        for i in range(self.num_planes):
            gva = sum([self.mat[i, j]*gv[j] for j in range(len(gv))])
            if gva not in Ideal(self.linear_forms()[i]):
                return False
        return True

    def compute_multi_minimal_generators(self):
        S1 = self.S**1
        M = []
        for alpha, m in zip(self.linear_forms(), self.multiplicity):
            if m == 0:
                continue
            RM = Sequence([module_elem(S1, (alpha.derivative(self.v[j]),))
                           for j in range(self.n)] + [module_elem(S1, (alpha**m,))])
            M.append(matrix([u[:-1] for u in syz(RM)]))
        return VectorFieldModule(list(module_intersection(M)))

    def free_resolution(self):
        M = self.minimal_generators()[0].v.parent()
        return M.submodule([g.v for g in self.minimal_generators()]).graded_free_resolution()

    def relations(self):
        mg = [g.v for g in self.minimal_generators()]
        return syz(Sequence([module_elem(self.S**self.n, tuple(mg[i]))
                             for i in range(len(mg))]))

    def euler_complement(self, g, alpha=None):
        if alpha is None:
            ag = self.v[-1]
        elif isinstance(alpha, Vector):  # alpha as a covector
            ag = self.euler().dot_product(alpha)
        elif isinstance(alpha, (int, Integer)):
            ag = self.v[int(alpha)]
        else:
            try:
                ag = self.v[int(alpha)]
            except Exception:
                ag = self.v[-1]
        return VectorField(g - ag(*tuple(g)) // ag * self.euler().v, self.S)

    def restriction(self, ind_or_H):
        B = copy(self.mat)
        if isinstance(ind_or_H, (int, Integer)):
            H = B[ind_or_H]
        elif isinstance(ind_or_H, list):
            H = vector(ind_or_H)
        else:
            H = ind_or_H
        # we will restrict to H
        piv = np.where(np.array(H) != 0)[0][0] # first non-zero coord
        for t in range(B.nrows()):
            B[t] = H[piv]*B[t] - B[t, piv]*H
        B = B.delete_columns([piv])#.delete_rows([i])
        B, multiplicity = remove_duplicate_planes(B)
        for t in range(B.nrows()):
            B[t] /= gcd(B[t])
            c = np.where(np.array(B[t]) != 0)[0][0]
            B[t] *= sgn(B[t, c])
        return HyperplaneArrangement(B, multiplicity=multiplicity)

    def localisation(self, L):
        I = Ideal(*[a.dot_product(self.euler().v) for a in self.mat[L, :]])
        indices = []
        for i, a in enumerate(self.mat):
            if a.dot_product(self.euler().v) in I:
                indices.append(i)
        return HyperplaneArrangement(self.mat[indices, :])

    def deletion(self, L):
        return HyperplaneArrangement(self.mat.delete_rows(L))

    def _coerce_hyperplane_indices(self, subset) -> Tuple[int, ...]:
        if isinstance(subset, (int, Integer)):
            indices = [int(subset)]
        else:
            try:
                indices = [int(i) for i in subset]
            except TypeError as exc:
                raise TypeError('subset must be an int or an iterable of ints.') from exc

        normalised = []
        seen = set()
        for index in indices:
            if index < 0 or index >= self.num_planes:
                raise IndexError(f'hyperplane index {index} is out of range for {self.num_planes} planes.')
            if index not in seen:
                normalised.append(index)
                seen.add(index)
        return tuple(sorted(normalised))

    def _subarrangement_from_indices(self, indices: Iterable[int]):
        rows = list(indices)
        multiplicity = None if self.multiplicity is None else [self.multiplicity[i] for i in rows]
        if not rows:
            return HyperplaneArrangement(matrix(self.K, 0, self.n), multiplicity=multiplicity)
        return HyperplaneArrangement(self.mat[rows, :], multiplicity=multiplicity)

    @cached_method
    def _hyperplanes_containing_rank_n_minus_1_flat(self, indices: Tuple[int, ...]) -> Tuple[int, ...]:
        if len(indices) != self.n - 1:
            raise ValueError(f'expected {self.n - 1} indices, got {len(indices)}.')

        submat = self.mat[list(indices), :]
        if submat.rank() != self.n - 1:
            return tuple()

        row_space = submat.row_space()
        return tuple(i for i, row in enumerate(self.mat.rows()) if row in row_space)

    def constructive_closure_indices(self, subset) -> Tuple[int, ...]:
        r"""Return the indices of the constructive closure ``\langle B \rangle_A``."""
        current = set(self._coerce_hyperplane_indices(subset))
        target_rank = self.n - 1

        if target_rank <= 0 or len(current) < target_rank:
            return tuple(sorted(current))

        while True:
            snapshot = tuple(sorted(current))
            updated = set(current)
            for flat_indices in itertools.combinations(snapshot, target_rank):
                updated.update(self._hyperplanes_containing_rank_n_minus_1_flat(flat_indices))
                if len(updated) == self.num_planes:
                    return tuple(range(self.num_planes))
            if updated == current:
                return tuple(sorted(current))
            current = updated

    def constructive_closure(self, subset, return_indices: bool = False):
        r"""Return the constructive closure ``\langle B \rangle_A`` as a subarrangement."""
        indices = self.constructive_closure_indices(subset)
        closure = self._subarrangement_from_indices(indices)
        if return_indices:
            return closure, indices
        return closure

    def constructively_generates(self, subset) -> bool:
        r"""Check whether ``subset`` constructively generates the full arrangement."""
        return self.constructive_closure_indices(subset) == tuple(range(self.num_planes))

    def minimal_constructive_subset_indices(self, max_size: Optional[int] = None) -> Tuple[int, ...]:
        r"""Return a minimum-cardinality subset ``B`` with ``A = \langle B \rangle_A``."""
        full_indices = tuple(range(self.num_planes))
        if self.num_planes == 0:
            return full_indices

        if self.n <= 2 or self.num_planes < self.n - 1:
            min_size = self.num_planes
        else:
            min_size = self.n - 1

        upper = self.num_planes if max_size is None else min(int(max_size), self.num_planes)
        if upper < min_size:
            raise ValueError(f'no constructively generating subset can have size <= {upper}.')

        for size in range(min_size, upper + 1):
            for subset in itertools.combinations(full_indices, size):
                if self.constructive_closure_indices(subset) == full_indices:
                    return subset

        raise ValueError(f'no constructively generating subset found up to size {upper}.')

    def minimal_constructive_subset(self, max_size: Optional[int] = None, return_indices: bool = False):
        r"""Return a minimum-cardinality constructive generating subarrangement."""
        indices = self.minimal_constructive_subset_indices(max_size=max_size)
        subset = self._subarrangement_from_indices(indices)
        if return_indices:
            return subset, indices
        return subset

    def s_invariant(self, max_size: Optional[int] = None) -> int:
        r"""Return ``s(A)``, the minimum size of a constructive generating subset."""
        return len(self.minimal_constructive_subset_indices(max_size=max_size))

    def addition(self, mat):
        if isinstance(mat, list):
            return HyperplaneArrangement(self.mat.stack(vector(mat)))
        else:
            return HyperplaneArrangement(self.mat.stack(mat))

    def intersection_lattice(self):
        r"""
        Return the intersection lattice of the arrangement as a Sage poset.

        Each lattice element is represented by a tuple of hyperplane indices:
        the indices of hyperplanes containing the corresponding flat.
        The order is set inclusion on these index tuples, which corresponds to
        reverse inclusion on geometric intersections.
        """
        from sage.combinat.posets.posets import Poset

        if self.num_planes == 0:
            return Poset([tuple()])

        rows = list(self.mat.rows())
        closure_cache = {tuple(): tuple()}
        rank_cache = {tuple(): 0}

        def closure(indices):
            key = tuple(sorted(indices))
            if key in closure_cache:
                return closure_cache[key]

            submat = self.mat[list(key), :]
            rk = submat.rank()
            if rk == 0:
                cl = tuple()
            else:
                rs = submat.row_space()
                cl = tuple(i for i, row in enumerate(rows) if row in rs)

            closure_cache[key] = cl
            rank_cache[cl] = rk
            return cl

        flats = {tuple()}
        queue = [tuple()]

        while queue:
            current = queue.pop()
            current_set = set(current)
            for i in range(self.num_planes):
                if i in current_set:
                    continue
                nxt = closure(current + (i,))
                if nxt not in flats:
                    flats.add(nxt)
                    queue.append(nxt)

        # Keep a deterministic order; rank is codimension in this representation.
        elements = tuple(sorted(flats, key=lambda f: (rank_cache.get(f, len(f)), len(f), f)))
        return Poset((elements, lambda a, b: set(a).issubset(set(b))))

    @cached_method
    def is_spog(self) -> Union[bool, List[int]]:
        fr = self.free_resolution()
        if fr._length != 2:
            return False
        if len(fr.betti(2)) != 1:
            return False

        level = -1
        for deg, multiplicity in fr.betti(2).items():
            if multiplicity != 1:
                return False
            else:
                level = deg - 1

        if level not in fr.betti(1).keys():
            return False

        degs = []
        for deg, multiplicity in fr.betti(1).items():
            if deg == level:
                degs.extend([deg]*(multiplicity - 1))
            else:
                degs.extend([deg]*multiplicity)
        return sorted(degs) + [level]

    def level_coeff(self):
        spog_res = self.is_spog()
        if not spog_res:
            raise ValueError('the given arrangement is not SPOG.')
        LHS = None
        RHS = []
        for g in self.relations():
            if 1 in [c.degree() for c in g]: # find degree one
                for i, c in enumerate(g):
                    if LHS is None and c.degree() == 1:
                        LHS = (c, i)  # c*MG[i]
                    elif c != 0:
                        RHS.append((-c, i)) # the relation
        return (LHS, RHS)

    def search_free_addition(self, a_range=range(-10, 10), b_range=range(-10, 10)):
        plane = None
        found = False
        try:
            from tqdm.auto import tqdm
        except ImportError:
            def tqdm(iterable, *args, **kwargs):
                return iterable
        for a in tqdm(a_range):
            if found:
                break
            for b in b_range:
                if found:
                    break
                for c in [0, 1]:
                    plane = [c, a, b] + [0]*(self.n - 3)
                    try:
                        B = self.addition(plane)
                    except Exception:
                        continue
                    if B.is_free:
                        print("plane: ", vector(plane), "degrees: ", B.degrees())
                        found = True
                        break
        if not found:
            print("No planes found")
        return plane

    def delta_I(self, I=None, generators=None):
        r"""
        Compute the signed maximal minors `\Delta_I` and Saito coefficients `g_I`.

        For a `p \times \ell` derivation matrix `M` with rows `\theta_1, \ldots, \theta_p`,
        and a subset `I = \{i_1 < \cdots < i_\ell\} \subseteq [p]` (0-indexed),
        define

        .. math::

            \Delta_I = (-1)^{\sigma(I)} \det(M_I), \qquad
            \sigma(I) = \sum_{k=1}^{\ell}(i_k - k) + \ell

        where `M_I` is the submatrix of rows indexed by `I`.
        Since each `\theta_i \in D(\mathcal{A})`, `\Delta_I` is divisible by `Q`,
        and we write `\Delta_I = g_I \cdot Q`.

        In the SPOG case (`p = \ell + 1`), we use the simplified notation
        `\Delta_i = (-1)^i \det(M_{[p] \setminus \{i\}})` (0-indexed `i`).

        Parameters
        ----------
        I : list of int or list of lists, optional
            A subset (or list of subsets) of row indices (0-indexed).
            If ``None``, all `\binom{p}{\ell}` subsets are computed.
        generators : list of VectorField, optional
            Custom generators to use instead of ``self.minimal_generators()``.

        Returns
        -------
        dict
            A dictionary mapping each tuple `I` to a pair `(\Delta_I, g_I)`.
        """
        if generators is not None:
            m = self._to_matrix(generators)
        else:
            m = self._to_matrix(self.minimal_generators())
        p = m.nrows()
        ell = self.n

        if I is not None:
            # allow a single subset
            if isinstance(I[0], (int, Integer)):
                subsets = [tuple(sorted(I))]
            else:
                subsets = [tuple(sorted(s)) for s in I]
        else:
            subsets = list(itertools.combinations(range(p), ell))

        result = {}
        for sub in subsets:
            sigma = sum(sub[k] - k for k in range(ell))
            sign = (-1) ** sigma
            det_val = sign * m[list(sub), :].det()
            g = det_val // self.Q
            result[sub] = (det_val, g)
        return result

    @staticmethod
    def _to_matrix(generators):
        """Convert generators to a matrix, accepting VectorField objects, vectors, or lists."""
        if hasattr(generators, 'gens'):  # VectorFieldModule
            return matrix([g.v for g in generators])
        rows = []
        for g in generators:
            if isinstance(g, VectorField):
                rows.append(g.v)
            elif hasattr(g, 'parent'):  # Sage vector
                rows.append(g)
            else:
                rows.append(vector(g))
        return matrix(rows)

    def saito_coefficients(self, generators=None, *, as_dict=False, signed=False):
        r"""
        Compute scaled maximal minors of a logarithmic derivation matrix.

        For a list ``G`` of ``p >= ell`` derivations, this computes
        ``det(M_I[G]) / Q(A)`` for every ``ell``-subset ``I`` of rows, in
        the sense of Definition ``def:scaled-minor-ideal``.

        For the historical SPOG case ``p = ell + 1``, the default return
        value remains the signed cofactor list
        ``[g_0, g_1, ..., g_ell]`` satisfying
        ``sum_i g_i theta_i = 0``.  Pass ``as_dict=True`` to get the
        definition-level dictionary for this case too.

        Parameters
        ----------
        generators : list, optional
            A list of VectorField objects, Sage vectors, or plain lists.
            If ``None``, uses ``self.minimal_generators()``.
        as_dict : bool
            If ``True``, return a dictionary mapping each row subset
            ``I`` to ``det(M_I) / Q(A)``.  If ``False`` and ``p=ell+1``,
            return the signed cofactor list.
        signed : bool
            Only used with ``as_dict=True``.  If ``True``, multiply each
            determinant by the same subset sign used by ``delta_I``.
            Signs are immaterial for the scaled minor ideal.

        Returns
        -------
        list or dict
            A signed list in the ``p=ell+1`` default case; otherwise a
            dictionary ``{I: det(M_I)/Q(A)}``.
        """
        if generators is not None:
            m = self._to_matrix(generators)
        else:
            m = self._to_matrix(self.minimal_generators())
        p = m.nrows()
        ell = self.n

        if p < ell:
            raise ValueError(f'saito_coefficients requires p >= ell, got p={p}, ell={ell}')

        if p == ell + 1 and not as_dict:
            coeffs = []
            for i in range(p):
                rows = [j for j in range(p) if j != i]
                det_val = (-1) ** i * m[rows, :].det()
                g = det_val // self.Q
                coeffs.append(g)

            # Verify the cofactor relation in the ell+1 case.
            relation = sum(coeffs[i] * m[i] for i in range(p))
            assert relation == 0, f"Relation check failed: {relation}"

            return coeffs

        coeffs = {}
        for sub in itertools.combinations(range(p), ell):
            det_val = m[list(sub), :].det()
            if signed:
                sign = (-1) ** sum(sub[k] - k for k in range(ell))
                det_val = sign * det_val
            g = det_val // self.Q
            coeffs[sub] = g

        return coeffs

    def scaled_minor_ideal(self, generators=None):
        r"""
        Return the scaled minor ideal ``I_A(G)``.

        For ``p >= ell`` candidate derivations, this is the ideal
        generated by ``det(M_I[G]) / Q(A)`` for all row subsets
        ``I`` with ``|I| = ell``.  If every scaled minor vanishes, the
        zero ideal is returned.
        """
        coeffs = self.saito_coefficients(generators, as_dict=True)
        coeffs = coeffs.values()
        nonzero = [c for c in coeffs if c != 0]
        if not nonzero:
            return self.S.ideal(self.S.zero())
        return self.S.ideal(nonzero)

    def scaled_minor_ideal_height(self, generators=None):
        r"""
        Return ``ht I_\mathcal{A}(G)``.

        Convention: if the ideal is improper (i.e. equal to ``S``), return
        ``+infinity`` to signal that the height hypothesis of
        Theorem 1.1 is violated by improperness rather than by low height.
        """
        I = self.scaled_minor_ideal(generators)
        if I.is_one():
            return float('inf')
        return self.n - I.dimension()

    def candidate_generates(self, generators):
        r"""
        Verify whether the given candidates generate ``D(\mathcal{A})``.

        Returns ``True`` iff every minimal generator of ``D(\mathcal{A})``
        is an ``S``-linear combination of ``generators``.  The check is
        performed via Singular's ``reduce`` against a standard basis of
        the submodule generated by the candidates.
        """
        from sage.all import singular as _sing
        _sing.eval('ring Rcg = 0,(' + ','.join(str(v) for v in self.v) + '),dp;')
        mod_body = ','.join(
            '[' + ','.join(str(c) for c in self._to_vector(g)) + ']'
            for g in generators
        )
        _sing.eval('module Mcg = ' + mod_body)
        _sing.eval('Mcg = std(Mcg)')
        for g in self.minimal_generators():
            v_str = '[' + ','.join(str(c) for c in g.v) + ']'
            red = _sing.eval('string(reduce(' + v_str + ', Mcg))')
            entries = red.strip().lstrip('[').rstrip(']').split(',')
            if not all(e.strip() == '0' for e in entries):
                return False
        return True

    @staticmethod
    def _to_vector(g):
        if isinstance(g, VectorField):
            return g.v
        if hasattr(g, 'parent'):
            return g
        return vector(g)

    def check_generalized_saito(self, generators=None, verify=True, verbose=True):
        r"""
        Check the hypotheses of "Saito-type generator criterion for nonfree arrangements".

        For ``\ell+1`` homogeneous derivations
        ``G = (\theta_1, \ldots, \theta_{\ell+1})`` in ``D(\mathcal{A})``:
        compute the signed scaled minors ``g_i`` and the ideal
        ``I_\mathcal{A}(G) = (g_1, \ldots, g_{\ell+1})``.  The criterion
        says: if ``I_\mathcal{A}(G)`` is proper and ``ht I_\mathcal{A}(G)
        \ge 3``, then ``G`` generates ``D(\mathcal{A})`` and its syzygy
        module is freely generated by ``(g_1,\ldots,g_{\ell+1})``.
        Additionally, if all ``g_i`` have positive degree and one is
        linear, ``G`` is minimal and ``\mathcal{A}`` is SPOG.

        Parameters
        ----------
        generators : iterable, optional
            ``\ell + 1`` candidate derivations.  If ``None``, use
            ``self.minimal_generators()``.
        verify : bool
            If ``True``, also independently check (by submodule equality)
            whether ``G`` actually generates ``D(\mathcal{A})``.
        verbose : bool
            Print a human-readable summary.

        Returns
        -------
        dict with keys
            ``coefficients``, ``ideal``, ``is_proper``, ``height``,
            ``criterion_applies``, ``all_positive_degree``,
            ``has_linear``, ``predicts_minimal_spog``,
            ``actually_generates`` (if ``verify=True``),
            ``counterexample`` (True if criterion applies but
            ``G`` does not generate ``D(\mathcal{A})`` -- which the
            theorem says should never happen).
        """
        if generators is None:
            gens = self.minimal_generators()
        else:
            gens = generators
        m = self._to_matrix(gens)
        if m.nrows() != self.n + 1:
            raise ValueError(
                f'check_generalized_saito requires ell + 1 generators, '
                f'got {m.nrows()} for ell={self.n}'
            )

        coeffs = self.saito_coefficients(gens)
        I = self.scaled_minor_ideal(gens)

        is_proper = not I.is_one()
        if is_proper:
            ht = self.n - I.dimension()
        else:
            ht = float('inf')

        criterion_applies = is_proper and ht >= 3

        nonzero = [c for c in coeffs if c != 0]
        all_pos_deg = bool(nonzero) and all(c.degree() > 0 for c in nonzero)
        has_linear = any(c.degree() == 1 for c in nonzero)
        predicts_minimal_spog = criterion_applies and all_pos_deg and has_linear

        # Check: g_1,...,g_ell have no common divisor mod g_{ell+1}
        # Compute gcd of g_rest modulo g_last
        ell = self.n
        g_last = coeffs[ell]
        g_rest = coeffs[:ell]
        I_mod = ideal(g_last)
        g_mod = [self.S(g).reduce(I_mod.groebner_basis()) for g in g_rest]
        common = gcd(g_mod)

        result = {
            'coefficients': coeffs,
            'ideal': I,
            'is_proper': is_proper,
            'height': ht,
            'criterion_applies': criterion_applies,
            'all_positive_degree': all_pos_deg,
            'has_linear': has_linear,
            'predicts_minimal_spog': predicts_minimal_spog,
            'common_divisor_mod': common,
        }

        if verify:
            actually = self.candidate_generates(gens)
            result['actually_generates'] = actually
            result['counterexample'] = criterion_applies and not actually

        if verbose:
            print(f"  scaled minors (g_i):")
            for i, g in enumerate(coeffs):
                print(f"    g_{i} = {g}  (deg {g.degree() if g != 0 else -1})")
            print(f"  proper:                     {is_proper}")
            print(f"  ht I_A(G):                  {ht}")
            print(f"  criterion applies (ht>=3):  {criterion_applies}")
            print(f"  all g_i of positive degree: {all_pos_deg}")
            print(f"  some g_i is linear:         {has_linear}")
            print(f"  predicts minimal SPOG:      {predicts_minimal_spog}")
            print(f"g_{{ell+1}} = {coeffs[ell]}")
            for i, g in enumerate(g_rest):
                print(f"  g_{i} = {g} (degree {g.degree() if g != 0 else -1})")
            print(f"  gcd(g_0,...,g_{{ell-1}}) mod g_{{ell}} = {common}")
            if verify:
                print(f"  actually generates D(A):    {result['actually_generates']}")
                if result['counterexample']:
                    print("  *** COUNTEREXAMPLE: criterion satisfied but G does not generate ***")

        return result

    def scaled_minor_tensor(self, generators=None, *, signed=True):
        r"""
        Return the alternating ``(p-ell)``-tensor of scaled maximal minors.

        For ``p = ell + k`` candidate derivations ``G`` with derivation
        matrix ``M``, this is the element

        .. math::

            \tilde g \;=\; \sum_{|T|=k}\,(-1)^{\sigma(T)}\,
                           g_{[p]\setminus T}\,e_{t_1}\wedge\cdots\wedge e_{t_k}
            \;\in\; \bigwedge^{k} S^{p},

        where ``g_I = det(M_I)/Q`` is the scaled ``ell``-minor and
        ``\sigma(T) = \sum_j t_j - j`` is the ``signed=True`` sign convention
        used by :meth:`saito_coefficients` and :meth:`delta_I`.

        With this convention, ``\tilde g`` satisfies the cofactor identity:
        contracting with the columns of ``M`` annihilates ``\tilde g`` in the
        sense that, for ``k = 2``, the antisymmetric matrix
        ``\tilde g`` (see :meth:`scaled_minor_matrix`) satisfies
        ``\tilde g \cdot M = 0`` identically.  More generally, every contraction
        of ``\tilde g`` with ``k - 1`` rows of ``M`` lies in the syzygy module
        of ``G``.

        Parameters
        ----------
        generators : iterable, optional
            Candidate derivations.  Defaults to ``self.minimal_generators()``.
        signed : bool
            If ``True`` (default), include the Hodge-star sign
            ``(-1)^{\sigma(T)}``.  If ``False``, return ``g_{[p]\setminus T}``
            with no extra sign.

        Returns
        -------
        dict
            ``{T: \tilde g_T}`` keyed by ``k``-subsets ``T \subset [p]``.
        """
        m = self._to_matrix(generators if generators is not None else self.minimal_generators())
        p = m.nrows()
        ell = self.n
        k = p - ell
        if k < 0:
            raise ValueError(f'scaled_minor_tensor requires p >= ell, got p={p}, ell={ell}')

        unsigned = self.saito_coefficients(generators=generators, as_dict=True, signed=False)

        tensor = {}
        for T in itertools.combinations(range(p), k):
            I = tuple(i for i in range(p) if i not in T)
            if signed:
                sigma_T = sum(T[j] - j for j in range(k))
                tensor[T] = (-1) ** sigma_T * unsigned[I]
            else:
                tensor[T] = unsigned[I]
        return tensor

    def scaled_minor_matrix(self, generators=None):
        r"""
        Antisymmetric ``p \times p`` matrix view of :meth:`scaled_minor_tensor`
        in the case ``p = ell + 2``.

        Entries: ``M[a, b] = (-1)^{a+b+1} g_{[p]\setminus\{a,b\}}`` for
        ``a < b`` and ``M[b, a] = -M[a, b]`` (zero diagonal).  Equivalently,
        ``M[a, b]`` is the signed scaled minor from
        :meth:`saito_coefficients` with ``as_dict=True, signed=True`` for the
        ``\ell``-subset ``[p] \setminus \{a, b\}``.

        Cofactor identity: ``self.scaled_minor_matrix(G) * M_G == 0`` holds
        identically (no hypotheses on ``G``), where ``M_G`` is the derivation
        matrix.  This is the reason the rows of ``\tilde g`` are
        automatically syzygies of ``G``.

        Raises ``ValueError`` unless ``p = ell + 2``.
        """
        m = self._to_matrix(generators if generators is not None else self.minimal_generators())
        p = m.nrows()
        ell = self.n
        if p != ell + 2:
            raise ValueError(
                f'scaled_minor_matrix requires p = ell + 2, got p={p}, ell={ell}'
            )

        tensor = self.scaled_minor_tensor(generators=generators, signed=True)
        mat = matrix(self.S, p, p)
        for T, value in tensor.items():
            a, b = T
            mat[a, b] = value
            mat[b, a] = -value
        return mat

    def plucker_relations(self, generators=None):
        r"""
        Evaluate the Grassmann--Plücker quadrics on :meth:`scaled_minor_tensor`.

        For each pair ``(T_1, T_2)`` with ``|T_1| = k + 1`` and ``|T_2| = k - 1``
        (where ``k = p - ell``), the Plücker quadric is

        .. math::

            Q(T_1, T_2) \;=\; \sum_{r=0}^{k}\,(-1)^r\,
                              \tilde g_{T_1\setminus\{j_r\}}\,
                              \tilde g_{T_2 \cup \{j_r\}}^{\rm signed}

        where ``T_1 = (j_0 < \cdots < j_k)`` and ``\tilde g_{T_2 \cup \{j_r\}}^{\rm signed}``
        denotes the signed component of ``\tilde g`` at the (canonically
        reordered) ``k``-subset ``T_2 \cup \{j_r\}`` (zero if ``j_r \in T_2``).

        ``\tilde g`` is **decomposable** (i.e., ``\tilde g = n_1 \wedge \cdots
        \wedge n_k`` for some ``n_i \in S^p``) iff every ``Q(T_1, T_2)``
        vanishes identically in ``S``.  See :meth:`is_plucker_decomposable`.

        For ``k \le 1`` the relations are vacuous (empty dict): every
        ``0``- or ``1``-tensor is decomposable.

        Returns
        -------
        dict
            ``{(T_1, T_2): Q(T_1, T_2)}`` of evaluated quadrics.
        """
        m = self._to_matrix(generators if generators is not None else self.minimal_generators())
        p = m.nrows()
        ell = self.n
        k = p - ell
        if k < 2:
            return {}

        tensor = self.scaled_minor_tensor(generators=generators, signed=True)

        def signed_component(T):
            """Signed component \\tilde g_T for any (possibly unordered or
            repeating) k-tuple T.  Returns 0 if T has duplicates; otherwise
            sorts T and multiplies by the sign of the permutation."""
            if len(set(T)) != len(T):
                return self.S.zero()
            T_list = list(T)
            sign = 1
            for i in range(len(T_list)):
                for j in range(i + 1, len(T_list)):
                    if T_list[i] > T_list[j]:
                        sign = -sign
            return sign * tensor[tuple(sorted(T_list))]

        relations = {}
        for T1 in itertools.combinations(range(p), k + 1):
            for T2 in itertools.combinations(range(p), k - 1):
                value = self.S.zero()
                for r, j in enumerate(T1):
                    T1_minus = tuple(t for t in T1 if t != j)
                    T2_plus = tuple(list(T2) + [j])
                    value += (-1) ** r * tensor[T1_minus] * signed_component(T2_plus)
                relations[(T1, T2)] = value
        return relations

    def is_plucker_decomposable(self, generators=None):
        r"""
        Return ``True`` iff the scaled minor tensor :meth:`scaled_minor_tensor`
        is decomposable, equivalently all :meth:`plucker_relations` vanish.

        For ``p \le ell + 1`` (``k \le 1``) the tensor is *automatically*
        decomposable, so this returns ``True`` unconditionally.
        """
        return all(v == 0 for v in self.plucker_relations(generators=generators).values())

    def unified_saito_diagnostic(self, generators=None, verify=True, verbose=True):
        r"""
        Report the data of the unified Saito-type setup for any ``p \ge ell``.

        This method computes — and, if ``verify=True``, cross-checks — the two
        $g_I$-only invariants that arise in the unified Saito-type criterion:

        (i)  the height ``ht I_A(G)`` of the scaled minor ideal, and

        (ii) whether the scaled minor tensor ``\tilde g \in \bigwedge^{p-\ell} S^p``
             is **Plücker-decomposable**, i.e., factors as
             ``\tilde g = n_1 \wedge \cdots \wedge n_k`` (cf.
             :meth:`is_plucker_decomposable`).

        For ``k = p - \ell \le 1`` the second invariant is automatic, and
        ``(i)`` reproduces:

        * ``k = 0``: the principal ideal ``\langle\det M / Q\rangle``.  The
          condition ``ht \ge 3`` forces this to be the unit ideal — Saito's
          classical theorem for free arrangements.
        * ``k = 1``: ``\tilde g \in S^p`` is a $1$-vector, always decomposable;
          the height bound ``\ge 3`` is the hypothesis of the generalised
          Saito / SPOG criterion (cf. :meth:`check_generalized_saito`).

        For ``k \ge 2``, decomposability is a non-trivial Grassmann--Plücker
        quadric condition on the ``g_I``.  **It is necessary that
        $\tilde g$ be decomposable for $G$ to generate $D(\mathcal{A})$** — the
        rows of the antisymmetric matrix view (``k = 2``) are syzygies of $G$
        by the cofactor identity, and S-spanning rows force decomposability.
        However, the conjunction "(i) and (ii)" is **not sufficient** for
        $G$ to generate $D(\mathcal{A})$; see ``Example 12`` of
        ``notebooks/Saito_criterion_examples.ipynb`` for an explicit
        ``\ell + 2``-generator counterexample.  The status of a complete
        $g_I$-only sufficient condition for $k \ge 2$ is open.

        Parameters
        ----------
        generators : iterable, optional
            Candidate derivations.  Defaults to ``self.minimal_generators()``.
        verify : bool
            If ``True``, also independently check (via :meth:`candidate_generates`)
            whether $G$ actually generates $D(\mathcal{A})$.
        verbose : bool
            Print a human-readable summary.

        Returns
        -------
        dict with keys
            ``p``, ``ell``, ``k``, ``height``, ``is_proper``, ``height_ok``
            (truth value of ``ht \ge 3`` or ``ht = \infty``), ``decomposable``
            (truth value of (ii)), ``necessary_conditions_hold`` (``True`` iff
            both ``height_ok`` and ``decomposable`` hold), and, if
            ``verify=True``, ``actually_generates`` and ``mismatch`` (``True``
            iff ``necessary_conditions_hold`` but ``G`` does not generate —
            i.e., a known instance of the gap between necessity and
            sufficiency).
        """
        gens = list(generators if generators is not None else self.minimal_generators())
        m = self._to_matrix(gens)
        p = m.nrows()
        ell = self.n
        if p < ell:
            raise ValueError(f'unified_saito_diagnostic requires p >= ell, got p={p}, ell={ell}')
        k = p - ell

        I_ideal = self.scaled_minor_ideal(gens)
        is_proper = not I_ideal.is_one()
        ht = self.scaled_minor_ideal_height(gens)
        height_ok = (ht == float('inf')) or (isinstance(ht, (int, Integer)) and ht >= 3)

        decomposable = self.is_plucker_decomposable(generators=gens)
        necessary = height_ok and decomposable

        result = {
            'p': p,
            'ell': ell,
            'k': k,
            'is_proper': is_proper,
            'height': ht,
            'height_ok': height_ok,
            'decomposable': decomposable,
            'necessary_conditions_hold': necessary,
        }

        if verify:
            actually = self.candidate_generates(gens)
            result['actually_generates'] = actually
            result['mismatch'] = necessary and not actually

        if verbose:
            print(f"  p = {p}, ell = {ell}, k = p - ell = {k}")
            print(f"  ht I_A(G):                       {ht}")
            print(f"  is_proper:                       {is_proper}")
            print(f"  (i)  ht >= 3 or improper:        {height_ok}")
            if k <= 1:
                print(f"  (ii) tilde g decomposable:       True (vacuous for k <= 1)")
            else:
                rels = self.plucker_relations(generators=gens)
                n_total = len(rels)
                n_zero = sum(1 for v in rels.values() if v == 0)
                print(f"  (ii) tilde g decomposable:       {decomposable}  "
                      f"({n_zero}/{n_total} Pluecker quadrics vanish)")
            print(f"  (i) and (ii) (necessary, not sufficient for k >= 2): {necessary}")
            if verify:
                print(f"  actually generates D(A):         {result['actually_generates']}")
                if result['mismatch']:
                    print("  >>> Necessary conditions hold but G does NOT generate;")
                    print("      this is a known gap between necessity and sufficiency for k >= 2.")

        return result

    def determinant_ideal(self, verbose=False):
        m = matrix([g.v for g in self.minimal_generators()])
        I = []
        for i in range(self.n - 1, m.nrows()):
            I0 = I.copy()
            for J in itertools.combinations(range(i), self.n - 1):
                determinant = m[J + (i,), :].det() // self.Q
                if verbose:
                    print(J + (i,))
                if determinant:
                    I.append(determinant)
                    if verbose:
                        print(latex(factor(determinant)))
                else:
                    if verbose:
                        print('dependent')
            if i > self.n - 1:
                if verbose:
                    print(f'{i}: increased?', Ideal(I0) != Ideal(I))
        return I

    def plot(self, u=None, xlim=None, ylim=None, Obs=None, ax=None, levels=0,
             quiver=True, cmap="coolwarm", nx=30, ny=30, scale=None, legend=True, offset=5):
        if self.n not in [2, 3]:
            raise ValueError('plot works only for two or three dimensional arrangements')

        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        latex_str = [latex(s) for s in self.linear_forms()]
        for i in range(self.num_planes):
            l = self._line_ends(self.mat[i], xlim=xlim, ylim=ylim)
            if l is not None:
                ax.axline(*l, label=f'${latex_str[i]}$')
                if legend:
                    ax.annotate(f'${latex_str[i]}$', l[1], xytext=(-offset, -1.5*offset),
                               textcoords='offset points')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        if levels > 0:
            px = np.linspace(xlim[0], xlim[1], max(nx, 100))
            py = np.linspace(ylim[0], ylim[1], max(ny, 100))
            xv_contour, yv_contour = np.meshgrid(px, py)
            Z = []
            if self.n == 3:
                for x, y in zip(xv_contour.ravel(), yv_contour.ravel()):
                    Z.append(self.Q(Rational(x), Rational(y), 1))
            elif self.n == 2:
                for x, y in zip(xv_contour.ravel(), yv_contour.ravel()):
                    Z.append(self.Q(Rational(x), Rational(y)))
            Z = np.array(Z).reshape(*xv_contour.shape)
            ax.contour(px, py, Z, levels=levels)

        if Obs is not None:
            points = np.array(list(Obs.keys()))
            vects = np.array(list(Obs.values()))
            ax.quiver(points[:, 0], points[:, 1], vects[:, 0], vects[:, 1], color='r')
            sing_points = points[(vects**2).sum(axis=1) < 1e-10]
            ax.scatter(sing_points[:, 0], sing_points[:, 1], marker='*', c='r')

        if u is not None:
            px = np.linspace(xlim[0], xlim[1], nx)
            py = np.linspace(ylim[0], ylim[1], ny)
            xv, yv = np.meshgrid(px, py)
            vx, vy = [], []

            u_v = u.v if isinstance(u, VectorField) else u

            if len(u_v) == 3:
                u_c = self.euler_complement(u_v, -1)
                u_c = u_c.v if isinstance(u_c, VectorField) else u_c
                for x, y in zip(xv.ravel(), yv.ravel()):
                    vx.append(u_c[0](Rational(x), Rational(y), 1))
                    vy.append(u_c[1](Rational(x), Rational(y), 1))
            elif len(u_v) == self.n - 1 and self.n == 3:
                u_c = u_v
                subs_template = {self.v[2]: 1}
                for x, y in zip(xv.ravel(), yv.ravel()):
                    subs_q = {self.v[0]: Rational(x), self.v[1]: Rational(y), **subs_template}
                    vx.append(u_v[0].subs(subs_q))
                    vy.append(u_v[1].subs(subs_q))
            else:
                u_c = u_v
                for x, y in zip(xv.ravel(), yv.ravel()):
                    vx.append(u_v[0](Rational(x), Rational(y)))
                    vy.append(u_v[1](Rational(x), Rational(y)))

            vx = np.array(vx).reshape(*xv.shape)
            vy = np.array(vy).reshape(*yv.shape)
            mgn = np.sqrt(vx*vx + vy*vy)

            if quiver:
                ax.quiver(xv, yv, vx, vy, mgn, cmap=cmap, scale=scale)
            else:
                ax.streamplot(xv, yv, vx, vy, color=mgn, linewidth=2, cmap=cmap,
                             density=[float(0.5), float(1.)])

            title = f'${latex(u_c)}$'
            if len(title) < 80 and legend:
                ax.set_title(title)
            else:
                ax.set_title("")
        else:
            if legend:
                Qstr = "".join([f"({x})" for x in latex_str])
                ax.set_title(f'${Qstr}$')

        ax.set_aspect('equal')
        ax.axis('off')
        return ax

    def plot_arr(self, **kwargs):
        return self.plot(u=None, **kwargs)

    def plot_vfield(self, u, **kwargs):
        return self.plot(u=u, **kwargs)

    def _line_ends(self,alpha,xlim=None,ylim=None):
        if xlim is None:
            xlim=(-1,1)
        if ylim is None:
            ylim=(-1,1)
        if len(alpha)==2:
            a,b = alpha
            c = 0
        else:
            a,b,c = alpha
        if (a==0 and b==0):
            return(None)
        if (a*b !=0) and ((-c-a*xlim[1])/b < ylim[0] or (-c-a*xlim[1])/b > ylim[1]):
            xlim = (xlim[0], -(b*ylim[0]+c)/a)
        return [(-c/a,q) if b==0 else (p,(-c-a*p)/b) for p,q in zip(xlim,(ylim[1],ylim[0]))]

    def list_ntf2(self, i):
        results = {}
        B = self.deletion([i])
        if B.is_free:
            print(f"del {i} is free", B.degrees())
        else:
            C = self.restriction(i)
            print(f"Res to {i} free?: {C.is_free}")
            for j in range(i + 1, self.num_planes):
                B = self.deletion([i, j])
                results[(i, j)] = B.degrees()
                if B.is_spog():
                    print('SPOG:', (i, j), B.degrees())
                elif B.is_free:
                    print('free: ', (i, j), B.degrees())
                else:
                    st = f"pd={len(B.free_resolution()) - 1}"
                    print(st, (i, j), B.degrees())
        return results

    def compute_basis(self, k0):
        mat = self.mat
        k = k0 - 1
        p = mat.nrows()
        n = mat.ncols()
        K = mat.base_ring()
        S = PolynomialRing(K, 'x', n)

        M, Sk1 = coef_map(k, S)
        m1 = binomial(n + k - 1, k)
        m2 = len(Sk1)

        C = zero_matrix(K, n*m2 + p*m1, p*m2)

        for i in range(p):
            D = zero_matrix(K, n*m2 + p*m1, m2)
            Y = sum(mat[i, j]*M[j] for j in range(n))
            D[(n*m2 + i*m1):(n*m2 + (i + 1)*m1), :] = Y
            for j in range(m2):
                D[j*n:(j + 1)*n, j] = mat[i, :].transpose()
            C[:, i*m2:(i + 1)*m2] = D

        B = C.sparse_matrix().left_kernel().matrix()
        B = B[:, :n*m2]

        basis = []
        for i in range(B.nrows()):
            if B[i] != 0:
                E = sum(Sk1[j]*B[i, j*n:(j + 1)*n] for j in range(m2))
                basis.append(VectorField(vector(E), S))

        return VectorFieldModule(basis)

    def compute_basis_linear(self, max_k=None, verbose=True):
        mat = self.mat
        p = mat.nrows()
        n = mat.ncols()
        max_k = p - n + 1 if max_k is None else max_k

        K = mat.base_ring()
        S = PolynomialRing(K, 'x', n)
        v = S.gens()

        V_mod = self.compute_basis(1)
        if verbose:
            print(f'deg 1, num gens {len(V_mod)}')

        MG = list(V_mod.gens)
        for k in range(max_k - 1):
            M = V_mod.image_lambda(n)
            V_mod = self.compute_basis(k + 2)

            Sk1_list = sk_expo(k + 2, n)
            Sk1_dic = {e: i for i, e in enumerate(Sk1_list)}
            Sk1 = [exponent_to_polynomial(expo, v) for expo in Sk1_list]

            V1 = V_mod.flatten_coefficients(Sk1_dic)
            lDk1 = M.augment(V1)
            piv = np.array(lDk1.pivots())
            mask = piv >= M.ncols()

            new_gens = [VectorField(vector(sum(Sk1[j]*lDk1[j*n:(j + 1)*n, i]
                                 for j in range(len(Sk1)))), S)
                       for i in piv[mask]]
            MG.extend(new_gens)

            if verbose:
                print(f'deg {k + 2}, dim {len(V_mod)}, '
                      f'codim {n*binomial(n + k, k + 1) - len(V_mod)}, '
                      f'num new gen {sum(mask)}')

        return VectorFieldModule(MG)

    def compute_basis_syzygy(self, verbose=True):
        mat = self.mat
        p = mat.nrows()
        n = mat.ncols()
        K = mat.base_ring()

        S = PolynomialRing(K, 'x', n)
        v = S.gens()
        PD = PolynomialRing(K, 'd', len(v))
        FM = PolynomialRing(K, list(v) + list(PD.gens()))
        d = FM.gens()[len(v):]

        if verbose:
            print(f'Number of planes: {p}')

        r = [d[i]*d[j] for i in range(len(v))
             for j in range(i, len(v))]

        GEN = []
        GEN_vec = []
        degs = []

        for k in range(p - n + 2):
            if sum(degs[:len(v) - 1]) + k < p < sum(degs):
                flag_skip = False
                for comb in itertools.combinations(degs, len(v) - 1):
                    if sum(comb) + k + 1 == p:
                        flag_skip = True
                        if verbose:
                            print(f"Skipping deg {k + 1} by Saito's criterion")
                        break
                if flag_skip:
                    continue

            G_mod = self.compute_basis(k + 1)
            if verbose:
                print(f'Number of vector generators at deg {k + 1}: {len(G_mod)}')

            I = Ideal(*r, *GEN)
            for g in G_mod:
                g_d = g.to_derivative()
                if g_d not in I:
                    GEN.append(g_d)
                    GEN_vec.append(g)
                    I = Ideal(*r, *GEN)
                    degs.append(k + 1)

                    if len(degs) == len(v) and sum(degs) == p:
                        if verbose:
                            print('Arrangement is free')
                            print('Degree sequence:', degs)
                        return VectorFieldModule(GEN_vec)

        if verbose:
            print('Degree sequence:', degs)
        return VectorFieldModule(GEN_vec)

    def fit_given_min_error(self, P, e0, verbose=True):
        from .fit import given_min_error as _given_min_error
        return _given_min_error(self, P, e0, verbose=verbose)
