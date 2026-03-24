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
    coord_vec,
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

        ## parameters
        self.multiplicity = multiplicity
        self.num_planes = self.mat.nrows() # number of hyperplanes in the arrangement

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
    def degs(self):
        return self.minimal_generators().degs()

    @property
    def is_free(self):
        return len(self.degs()) == self.n

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
        for eq in hull.equations:
            coeffs = [HyperplaneArrangement._coerce_vertex_scalar(val, base_field, max_denominator)
                      for val in eq[:-1]]
            coeffs.append(HyperplaneArrangement._coerce_vertex_scalar(eq[-1], base_field, max_denominator))
            coeffs = HyperplaneArrangement._normalise_plane_row(coeffs, base_field)
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
            RM = Sequence([module_elem(S1, (alpha.derivative(self.v[j]),))
                           for j in range(self.n)] + [module_elem(S1, (alpha**m,))])
            M.append(matrix([u[:-1] for u in syz(RM)]))
        return list(module_intersection(M))

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

    def addition(self, mat):
        if isinstance(mat, list):
            return HyperplaneArrangement(self.mat.stack(vector(mat)))
        else:
            return HyperplaneArrangement(self.mat.stack(mat))

    def intersection_lattice(self):
        return None

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
                        print("plane: ", vector(plane), "degrees: ", B.degs())
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

    def saito_coefficients(self, generators=None):
        r"""
        For the SPOG case (`p = \ell + 1`), compute the simplified Saito coefficients.

        Returns a list `[g_0, g_1, \ldots, g_\ell]` where
        `\Delta_i = (-1)^i \det(M_{[p] \setminus \{i\}}) = g_i \cdot Q`.

        Also verifies the relation `\sum g_i \theta_i = 0`.

        Parameters
        ----------
        generators : list, optional
            A list of VectorField objects, Sage vectors, or plain lists.
            If ``None``, uses ``self.minimal_generators()``.

        Returns
        -------
        list
            The coefficients `[g_0, \ldots, g_\ell]`.
        """
        if generators is not None:
            m = self._to_matrix(generators)
        else:
            m = self._to_matrix(self.minimal_generators())
        p = m.nrows()
        ell = self.n

        if p != ell + 1:
            raise ValueError(f'saito_coefficients requires p = ell + 1, got p={p}, ell={ell}')

        coeffs = []
        for i in range(p):
            rows = [j for j in range(p) if j != i]
            det_val = (-1) ** i * m[rows, :].det()
            g = det_val // self.Q
            coeffs.append(g)

        # verify relation
        relation = sum(coeffs[i] * m[i] for i in range(p))
        assert relation == 0, f"Relation check failed: {relation}"

        return coeffs

    def check_saito_criterion(self, generators=None, verbose=True):
        r"""
        Check the conditions of the generalized Saito criterion (Theorem 1 in saito.tex).

        For `\ell + 1` generators, checks:
        1. `g_{\ell+1} \in S_1 \setminus \{0\}` (i.e., degree 1 and nonzero)
        2. `g_1, \ldots, g_\ell \in S_{>0}` have no non-trivial common divisor modulo `g_{\ell+1}`

        Parameters
        ----------
        generators : list, optional
            A list of VectorField objects, Sage vectors, or plain lists.
            If ``None``, uses ``self.minimal_generators()``.

        Returns
        -------
        dict with keys 'coefficients', 'is_spog_by_criterion', 'g_last_deg', 'common_divisor_mod'
        """
        coeffs = self.saito_coefficients(generators)
        ell = self.n
        g_last = coeffs[ell]
        g_rest = coeffs[:ell]

        g_last_deg = g_last.degree() if g_last != 0 else -1
        cond1 = (g_last != 0 and g_last_deg == 1)

        # Check: g_1,...,g_ell have no common divisor mod g_{ell+1}
        # Compute gcd of g_rest modulo g_last
        I_mod = ideal(g_last)
        g_mod = [self.S(g).reduce(I_mod.groebner_basis()) for g in g_rest]
        common = gcd(g_mod)
        cond2 = (common.degree() == 0)  # common divisor is a constant

        result = {
            'coefficients': coeffs,
            'is_spog_by_criterion': cond1 and cond2,
            'g_last_deg': g_last_deg,
            'common_divisor_mod': common,
        }

        if verbose:
            print(f"g_{{ell+1}} = {coeffs[ell]}, degree = {g_last_deg}")
            print(f"  Condition 1 (deg=1, nonzero): {cond1}")
            for i, g in enumerate(g_rest):
                print(f"  g_{i} = {g} (degree {g.degree() if g != 0 else -1})")
            print(f"  gcd(g_0,...,g_{{ell-1}}) mod g_{{ell}} = {common}")
            print(f"  Condition 2 (no common divisor mod g_{{ell}}): {cond2}")
            print(f"  => SPOG by criterion: {cond1 and cond2}")

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


def list_ntf2(i, A=None):
    results = {}
    B = A.deletion([i])
    if B.is_free:
        print(f"del {i} is free", B.degs())
    else:
        C = A.restriction(i)
        print(f"Res to {i} free?: {C.is_free}")
        for j in range(i + 1, A.num_planes):
            B = A.deletion([i, j])
            results[(i, j)] = B.degs()
            if B.is_spog():
                print('SPOG:', (i, j), B.degs())
            elif B.is_free:
                print('free: ', (i, j), B.degs())
            else:
                st = f"pd={len(B.free_resolution()) - 1}"
                print(st, (i, j), B.degs())
    return results

def basis_da(mat, k0):
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

def min_gen_arr_linear(mat, max_k=None, verbose=True):
    p = mat.nrows()
    n = mat.ncols()
    max_k = p - n + 1 if max_k is None else max_k

    K = mat.base_ring()
    S = PolynomialRing(K, 'x', n)
    v = S.gens()

    V_mod = basis_da(mat, 1)
    if verbose:
        print(f'deg 1, num gens {len(V_mod)}')

    MG = list(V_mod.gens)
    for k in range(max_k - 1):
        M = V_mod.image_lambda(n)
        V_mod = basis_da(mat, k + 2)

        Sk1_list = sk_expo(k + 2, n)
        Sk1_dic = {e: i for i, e in enumerate(Sk1_list)}
        Sk1 = [exponent_to_polynomial(expo, v) for expo in Sk1_list]

        V1 = V_mod.to_flatten(Sk1_dic)
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

def min_gen_arr(mat, verbose=True):
    if not is_distinct_planes(mat):
        raise ValueError('The arrangement contains duplicated planes!')

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

        G_mod = basis_da(mat, k + 1)
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



HyperPlaneArr = HyperplaneArrangement

# Shim functions for backward compatibility
def vector_to_derivative(u):
    from .vector_field import VectorField
    return VectorField(u).to_derivative()

def derivative_to_vector(u):
    from .vector_field import VectorField
    return VectorField.from_derivative(u).v
    
def div(u, S):
    from .vector_field import VectorField
    return VectorField(u, S).div()
    
def rot(u, S):
    from .vector_field import VectorField
    return VectorField(u, S).rot()
    
def laplacian(u, S):
    from .vector_field import VectorField
    return VectorField(u, S).laplacian().v

def degseq(MG):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(MG).degs()

def is_s_indep(MG):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(MG).is_s_indep()
    
def saito(MG):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(MG).saito()
    
def vector_basis(G):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(G).vector_basis().gens
    
def vector_to_flatten(U, Sk_dic, n=0):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(U).to_flatten(Sk_dic, n)
    
def graded_component(G, deg):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(G).graded_component(deg).gens
    
def gendic(generators):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(generators).gendic()
    
def minimal_generating_set(G):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(G).minimal_generating_set().gens
    
def image_lambda(V, n):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(V).image_lambda(n)

def divergence_free(G):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(G).divergence_free().gens

def rotation_free(G):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(G).rotation_free().gens

def harmonic(G):
    from .vector_field import VectorFieldModule
    return VectorFieldModule(G).harmonic().gens


def dehomogenise(A, G):
    from .vector_field import VectorFieldModule
    if isinstance(G, VectorFieldModule):
        return G.dehomogenise().gens
    return VectorFieldModule(G).dehomogenise().gens
    
def affine_basis(A, G):
    from .vector_field import VectorFieldModule
    if isinstance(G, VectorFieldModule):
        return G.affine_basis().gens
    return VectorFieldModule(G).affine_basis().gens
    
def fit_vf(A, Obs, mod_gens, verbose=True):
    from .fit import fit_vf as _fit_vf
    return _fit_vf(A, Obs, mod_gens, verbose)
    
def fit_vorticity(A, Obs, mod_gens, verbose=True):
    from .fit import fit_vorticity as _fit_vorticity
    return _fit_vorticity(A, Obs, mod_gens, verbose)
    
def given_min_error(A, P, e0, verbose=True):
    from .fit import given_min_error as _given_min_error
    return _given_min_error(A, P, e0, verbose)

__all__ = [
    'HyperplaneArrangement',
    'HyperPlaneArr',
    'VectorField',
    'VectorFieldModule',
    'degseq',
    'is_distinct_planes',
    'remove_duplicate_planes',
    'coord_vec',
    'is_s_indep',
    'saito',
    'vector_basis',
    'list_ntf2',
    'create_generic_arrangement',
    'vector_to_flatten',
    'graded_component',
    'vector_to_derivative',
    'derivative_to_vector',
    'gendic',
    'module_intersection',
    'sk_expo',
    'exponent_to_polynomial',
    'coef_map',
    'image_lambda',
    'basis_da',
    'min_gen_arr_linear',
    'minimal_generating_set',
    'min_gen_arr',
    'dehomogenise',
    'affine_basis',
    'fit_vf',
    'fit_vorticity',
    'given_min_error',
    'div',
    'rot',
    'laplacian',
    'divergence_free',
    'rotation_free',
    'harmonic',
]
