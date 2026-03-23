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

# Singular functions
minbase = lambda L: singular_function('minbase')(Sequence(L))
singular_lib('presolve.lib')
syz = singular_function("syz")
degreepart = lambda G, deg: singular_function("degreepart")(Sequence([module_elem(G[0].parent(), list(G[j])) for j in range(len(G))]), deg, deg)
kbase = lambda G, deg: singular_function("kbase")(singular_function("std")(Sequence([module_elem(G[0].parent(), list(G[j])) for j in range(len(G))])), deg)
groebner = singular_function("groebner")

ideal = Ideal


class HyperplaneArrangement(SageObject):
    r"""
    The main class for the logarithmic derivation module of a central arrangement.

    INPUT:

    - ``mat`` -- matrix or list; a `p \times n` matrix whose row vectors correspond to the linear forms defining the planes
    - ``vertices`` -- iterable of vertex coordinates; used to build the convex hull and derive its supporting hyperplanes automatically (homogenised for a central arrangement)
    - ``Q`` -- polynomial (default: `None`); the defining polynomial of an arrangement that is the product of all linear forms defining the planes
    - ``base_field`` -- field (default: `QQ`); Base field for the ambient space. Used only when the arrangement is specified by a list or an array
    - ``vertex_max_denominator`` -- positive integer (default: `10^6`); bound for rational approximation when vertices contain floating point numbers
    - ``multiplicity`` -- list of integers (default: `None`); multiplicities of the hyperplanes

    ATTRIBUTES:

    - ``mat``: a matrix whose row vectors correspond to the linear forms defining the planes
    - ``K``: the base field
    - ``n``: the dimension of the ambient space V
    - ``v``: the generators [v[0],...,v[n-1]] of V^*
    - ``S``: the base polynomial ring K[v[0],...,v[n-1]]
    - ``Q``: the defining polynomial
    - ``jacob_I``: the Jacobian ideal of Q
    - ``minimal_generators``: the list of minimal generators of the logarithmic derivation module D(A)
    - ``euler``: the Euler derivation
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
        return vector(self.v)

    @cached_method
    def jacob_I(self):
        return ideal(*[self.Q.derivative(self.v[j]) for j in range(self.n)])

    @cached_method
    def minimal_generators(self):
        # compute minimal generators of D(A) by D(A) \cong <Euler> \oplus Syz(J_Q)
        return [self.euler()] + [self.euler_complement(u) for u in minbase(self.jacob_I().syzygy_module())]

    @cached_method
    def degs(self):
        return degseq(self.minimal_generators())

    @property
    def is_free(self):
        return len(self.degs()) == self.n

    @cached_method
    def linear_forms(self):
        return [sum(self.mat[i, j]*self.v[j] for j in range(self.n)) for i in range(self.num_planes)]

    @staticmethod
    def _matrix_from_vertices(vertices, *, base_field=QQ, max_denominator=10**6):
        r"""
        Build a homogenised matrix of supporting planes from vertex coordinates.

        INPUT:

        - ``vertices`` -- iterable of coordinate iterables describing the vertices
          of a convex polytope embedded in ``RR^d``.
        - ``base_field`` -- base field used to store the coefficients; defaults to
          ``QQ``.
        - ``max_denominator`` -- positive integer that bounds the denominator used
          when rationalising floating-point coordinates.

        OUTPUT:

        A matrix whose rows encode the linear forms of the supporting hyperplanes
        of the convex hull of ``vertices``.

        """
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

        # Include the hyperplane at infinity x_0 = 0 so that the cone
        # construction matches Q = x_0 * prod(h_i) as required by the
        # logarithmic derivation theory.  Without this extra plane the
        # dehomogenisation / euler_complement pipeline cannot guarantee
        # affine tangency.
        n_homog = len(rows[0])            # dim + 1 (spatial coords + constant)
        h_inf = [base_field(0)] * (n_homog - 1) + [base_field(1)]
        rows.insert(0, h_inf)

        return matrix(base_field, rows)

    @staticmethod
    def _coerce_vertex_scalar(value, base_field, max_denominator):
        r"""
        Coerce a numeric coefficient into ``base_field`` using Sage conventions.

        INPUT:

        - ``value`` -- number-like object arising from the hull computation.
        - ``base_field`` -- field that will store the coefficient.
        - ``max_denominator`` -- denominator bound when ``base_field`` is ``QQ``.

        OUTPUT:

        Element of ``base_field`` representing ``value``.

        """
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
        r"""
        Scale plane coefficients to a primitive integer vector when possible.

        INPUT:

        - ``coeffs`` -- list of coefficients describing a supporting hyperplane.
        - ``base_field`` -- field of the arrangement; only ``QQ`` triggers
          rescaling.

        OUTPUT:

        Normalised coefficient list whose first non-zero entry is positive.

        """
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

        # Fix orientation so first non-zero coefficient is positive
        sign = 1
        for num in numerators:
            if num != 0:
                sign = 1 if num > 0 else -1
                break

        return [base_field(sign * num) for num in numerators]

    def is_in_DA(self, gv):
        r"""
        Test whether a vector field lies inside the logarithmic module ``D(A)``.

        INPUT:

        - ``gv`` -- element of ``S^n`` to be tested.

        OUTPUT:

        Boolean indicating membership in ``D(A)``.

        """
        for i in range(self.num_planes):
            gva = sum([self.mat[i, j]*gv[j] for j in range(len(gv))])
            if gva not in Ideal(self.linear_forms()[i]):
                return False
        return True

    def compute_multi_minimal_generators(self):
        r"""
        Compute minimal generators for a multi-arrangement restriction.

        OUTPUT:

        List of module generators describing the logarithmic derivations for the
        multi-arrangement.

        """
        S1 = self.S**1
        M = []
        for alpha, m in zip(self.linear_forms(), self.multiplicity):
            RM = Sequence([module_elem(S1, (alpha.derivative(self.v[j]),))
                           for j in range(self.n)] + [module_elem(S1, (alpha**m,))])
            M.append(matrix([u[:-1] for u in syz(RM)]))
        return list(module_intersection(M))

    def free_resolution(self):
        r"""Return the graded free resolution of ``D(A)``."""
        M = self.minimal_generators()[0].parent()
        return M.submodule(self.minimal_generators()).graded_free_resolution()

    def relations(self):
        r"""Compute syzygies among the minimal generators of ``D(A)``."""
        mg = self.minimal_generators()
        return syz(Sequence([module_elem(self.S**self.n, tuple(mg[i]))
                             for i in range(len(mg))]))

    def euler_complement(self, g, alpha=None):
        r"""
        Project a vector field onto the complement of the Euler derivation.

        INPUT:

        - ``g`` -- element of ``S^n``.
        - ``alpha`` -- optional covector or index specifying which coordinate to
          eliminate; defaults to the last coordinate.

        OUTPUT:

        Element of ``D_0(A)`` representing the projection of ``g``.

        """
        if alpha is None:
            ag = self.v[-1]
        elif isinstance(alpha, Vector):  # alpha as a covector
            ag = self.euler().dot_product(alpha)
        elif isinstance(alpha, (int, Integer)):
            ag = self.v[int(alpha)]
        else:
            # Fallback: try to treat alpha as an index
            try:
                ag = self.v[int(alpha)]
            except Exception:
                ag = self.v[-1]
        return (g - ag(*g) // ag * self.euler())

    def restriction(self, ind_or_H):
        r"""
        Restrict the arrangement with respect to a hyperplane.

        INPUT:

        - ``ind_or_H`` -- either an index or an explicit hyperplane vector.

        OUTPUT:

        ``HyperplaneArrangement`` describing the induced multi-arrangement.

        """
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
        r"""
        Localise the arrangement at the intersection indexed by ``L``.

        INPUT:

        - ``L`` -- list of hyperplane indices.

        OUTPUT:

        ``HyperplaneArrangement`` consisting of all planes containing the
        intersection ``\cap_{i in L} A_i``.

        """
        I = Ideal(*[a.dot_product(self.euler()) for a in self.mat[L, :]])
        indices = []
        for i, a in enumerate(self.mat):
            if a.dot_product(self.euler()) in I:
                indices.append(i)
        return HyperplaneArrangement(self.mat[indices, :])

    def deletion(self, L):
        r"""
        Delete hyperplanes with indices in ``L``.

        OUTPUT:

        ``HyperplaneArrangement`` with the specified rows removed.

        """
        return HyperplaneArrangement(self.mat.delete_rows(L))

    def addition(self, mat):
        r"""Return the arrangement obtained by adding extra planes."""
        if isinstance(mat, list):
            return HyperplaneArrangement(self.mat.stack(vector(mat)))
        else:
            return HyperplaneArrangement(self.mat.stack(mat))

    def intersection_lattice(self):
        r"""(Placeholder) return the intersection lattice of the arrangement."""
        return None

    @cached_method
    def is_spog(self) -> Union[bool, List[int]]:
        r"""
        Decide whether the arrangement is strictly plus-one generated (SPOG).

        OUTPUT:

        ``False`` if the arrangement is not SPOG; otherwise the exponent list
        with the level appended.

        """
        fr = self.free_resolution()
        if fr._length != 2:
            return False
        if len(fr.betti(2)) != 1: # there should be only a single relation
            return False

        # relations
        level = -1
        for deg, multiplicity in fr.betti(2).items():
            if multiplicity != 1:
                return False
            else:
                level = deg - 1

        if level not in fr.betti(1).keys(): # the relation should be in degree level+1
            return False

        degs = []
        for deg, multiplicity in fr.betti(1).items(): # generators
            if deg == level:
                degs.extend([deg]*(multiplicity - 1))
            else:
                degs.extend([deg]*multiplicity)
        return sorted(degs) + [level]

    def level_coeff(self):
        r"""Return the level coefficient and level element for SPOG arrangements."""
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

    # search for a free addition: experimental
    def search_free_addition(self, a_range=range(-10, 10), b_range=range(-10, 10)):
        r"""
        Exhaustively search for free additions within the provided ranges.

        INPUT:

        - ``a_range`` -- iterable of integers for the second coordinate.
        - ``b_range`` -- iterable of integers for the third coordinate.

        OUTPUT:

        Vector describing a plane that yields a free addition, if found.

        """
        plane = None
        found = False
        try:
            from tqdm.auto import tqdm
        except ImportError:
            # Fallback if tqdm is not available
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

    def determinant_ideal(self, verbose=False):
        r"""Compute the determinant ideal associated with ``D(A)``."""
        m = matrix(self.minimal_generators())
        # m = m*random_matrix(QQ,n,n) # linear transformation does not change the ideal

        I = []
        for i in range(self.n - 1, m.nrows()):
            I0 = I.copy()
            for J in itertools.combinations(range(i), self.n - 1):
                determinant = m[J + (i,), :].det() // self.Q ## for simplicity, we devide by Q
                if verbose:
                    print(J + (i,))
                if determinant:
                    I.append(determinant)
                    if verbose:
                        print(latex(factor(determinant)))
                else:
                    if verbose:
                        print('dependent')
            if i > self.n - 1: # check if every addition on a new generator increases the ideal(det)
                # print(I0,I)
                if verbose:
                    print(f'{i}: increased?', Ideal(I0) != Ideal(I))
        return I
        # return(ideal(I))

    def plot(self, u=None, xlim=None, ylim=None, Obs=None, ax=None, levels=0,
             quiver=True, cmap="coolwarm", nx=30, ny=30, scale=None, legend=True, offset=5):
        r"""
        Plot the hyperplane arrangement and optionally a vector field.

        INPUT:

        - ``u`` -- (optional) vector field in ``D(A)`` to plot
        - ``xlim``/``ylim`` -- optional plotting ranges
        - ``Obs`` -- dictionary of observations for plotting vector data
        - ``ax`` -- optional matplotlib axis to plot on
        - ``levels`` -- number of contour levels of ``Q`` to render
        - ``quiver`` -- (default: True) whether to plot arrows instead of streamlines for vector field
        - ``cmap`` -- (default: "coolwarm") colormap for vector field
        - ``nx``/``ny`` -- grid resolution for vector field (default: 30)
        - ``scale`` -- optional scale parameter for quiver plot
        - ``legend`` -- (default: True) whether to show legend
        - ``offset`` -- (default: 5) offset for hyperplane labels

        OUTPUT:

        Matplotlib axis containing the rendered plot

        EXAMPLES::

            sage: A = HyperplaneArrangement(matrix(QQ, [[1,0], [0,1], [1,1]]))
            sage: A.plot()  # Plot just the arrangement
            sage: A.plot(u=A.euler())  # Plot with Euler vector field
        """
        if self.n not in [2, 3]:
            raise ValueError('plot works only for two or three dimensional arrangements')

        # Create axis if not provided
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.set_aspect('equal')
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)

        # Plot hyperplanes
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

        # Plot contours of Q
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

        # Plot predefined observation vectors
        if Obs is not None:
            points = np.array(list(Obs.keys()))
            vects = np.array(list(Obs.values()))
            ax.quiver(points[:, 0], points[:, 1], vects[:, 0], vects[:, 1], color='r')
            sing_points = points[(vects**2).sum(axis=1) < 1e-10]
            ax.scatter(sing_points[:, 0], sing_points[:, 1], marker='*', c='r')

        # Plot vector field if provided
        if u is not None:
            px = np.linspace(xlim[0], xlim[1], nx)
            py = np.linspace(ylim[0], ylim[1], ny)
            xv, yv = np.meshgrid(px, py)
            vx, vy = [], []

            if len(u) == 3:
                u_c = self.euler_complement(u, -1)
                for x, y in zip(xv.ravel(), yv.ravel()):
                    vx.append(u_c[0](Rational(x), Rational(y), 1))
                    vy.append(u_c[1](Rational(x), Rational(y), 1))
            elif len(u) == self.n - 1 and self.n == 3:
                u_c = u
                subs_template = {self.v[2]: 1}
                for x, y in zip(xv.ravel(), yv.ravel()):
                    subs_q = {self.v[0]: Rational(x), self.v[1]: Rational(y), **subs_template}
                    vx.append(u[0].subs(subs_q))
                    vy.append(u[1].subs(subs_q))
            else:
                u_c = u
                for x, y in zip(xv.ravel(), yv.ravel()):
                    vx.append(u[0](Rational(x), Rational(y)))
                    vy.append(u[1](Rational(x), Rational(y)))

            vx = np.array(vx).reshape(*xv.shape)
            vy = np.array(vy).reshape(*yv.shape)
            mgn = np.sqrt(vx*vx + vy*vy)

            if quiver:
                ax.quiver(xv, yv, vx, vy, mgn, cmap=cmap, scale=scale)
            else:
                ax.streamplot(xv, yv, vx, vy, color=mgn, linewidth=2, cmap=cmap,
                             density=[float(0.5), float(1.)])

            # Update title for vector field
            title = f'${latex(u_c)}$'
            if len(title) < 80 and legend:
                ax.set_title(title)
            else:
                ax.set_title("")
        else:
            # Title for arrangement only
            if legend:
                Qstr = "".join([f"({x})" for x in latex_str])
                ax.set_title(f'${Qstr}$')

        ax.set_aspect('equal')
        ax.axis('off')
        return ax

    def plot_arr(self, **kwargs):
        r"""Backward-compatible wrapper for plotting only the arrangement."""
        return self.plot(u=None, **kwargs)

    def plot_vfield(self, u, **kwargs):
        r"""Backward-compatible wrapper for plotting a vector field on ``self``."""
        return self.plot(u=u, **kwargs)

    def _line_ends(self,alpha,xlim=None,ylim=None):
        # find end points of the line defined by a linear form alpha
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


# degree sequence of list of vector fields
def degseq(MG):
    r"""Return the sorted degree sequence of a list of derivations."""
    return sorted([max(m.degree() for m in g) for g in MG])

# check if all the planes are distinct
def is_distinct_planes(mat):
    r"""Check whether the rows of ``mat`` define distinct hyperplanes."""
    for plane in itertools.combinations(range(mat.nrows()), 2):
        if mat[plane, :].rank() < 2:
            return False
    return True

# remove duplicate planes
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

# coordinate vectors
def coord_vec(n):
    r"""Return the list of coordinate vectors in ``QQ^n``."""
    return list(np.eye(n))

## check if elements of S^n are S-independent
def is_s_indep(MG):
    r"""Check S-linear independence via Saito's criterion."""
    return saito(MG).det() != 0

## The matrix in Saito's criterion
def saito(MG):
    r"""Return the matrix used in Saito's criterion for freeness."""
    assert len(MG[0]) == len(MG)
    M = matrix(MG)
    return M

# find linearly independent derivations among a given list G
def vector_basis(G):
    r"""Extract a vector-space basis from a list of derivations."""
    if not G:
        return []
    n_vars = len(G[0].base_ring().gens())
    n_comp = len(G[0])
    maxdeg = degseq(G)[-1]
    if maxdeg < 1:
        return [G[0]]
    Sk_list = []
    for k in range(maxdeg + 1):
        Sk_list.extend(sk_expo(k, n_vars))
    Sk_dic = {e: i for i, e in enumerate(Sk_list)}
    if len(Sk_dic) == 0:
        return []
    V1 = vector_to_flatten(G, Sk_dic, n=n_comp)
    return [G[i] for i in V1.pivots()]

# experimental
def list_ntf2(i, A=None):
    r"""Diagnostic helper exploring deletions and restrictions of ``A``."""
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


# ugly way to produce a generic arrangement
def create_generic_arrangement(n, k):
    r"""Return a random generic arrangement in ``QQ^n`` with ``k`` planes."""
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

def vector_to_flatten(U, Sk_dic, n=0):
    """
    Convert a list of elements in S^n to flattened vectors.

    INPUT:

    - ``U`` -- list of elements in S^n
    - ``Sk_dic`` -- dictionary mapping exponents to indices
    - ``n`` -- (optional) number of variables; if 0, determined from Sk_dic

    OUTPUT:

    Matrix whose columns are the flattened vectors

    EXAMPLES::

        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: U = [vector([x, y])]
        sage: Sk_dic = {(1,0): 0, (0,1): 1}
        sage: M = vector_to_flatten(U, Sk_dic)
        sage: M.dimensions()
        (4, 1)
    """
    if not U:
        raise ValueError("Input list U cannot be empty")

    # Determine number of generators
    ngens = len(next(iter(Sk_dic)))
    nn = ngens if n == 0 else n

    # Build matrix entries
    L = {}
    for j, u in enumerate(U):
        for i in range(nn):
            uu = u if nn == 1 else u[i]
            try:
                for e, c in uu.dict().items():
                    L[Sk_dic[tuple(e[:ngens])]*nn + i, j] = c
            except (KeyError, AttributeError) as err:
                raise ValueError(f"Invalid polynomial terms found: {err}")

    # Create sparse matrix
    return MatrixArgs(QQ, len(Sk_dic)*nn, len(U), entries=L).matrix()

def graded_component(G, deg):
    """
    Compute the graded component of degree deg of the module generated by G.

    INPUT:

    - ``G`` -- list of generators
    - ``deg`` -- target degree

    OUTPUT:

    List of basis elements of the graded component

    EXAMPLES::

        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: G = [vector([x, 0]), vector([0, y])]
        sage: comp = graded_component(G, 2)
        sage: len(comp)  # dimension of degree 2 component
        3
    """
    if not G:
        return []

    S = G[0].parent().base_ring()
    v = S.gens()

    result = []
    for d, u in zip(degseq(G), G):
        if d <= deg:
            # Generate monomials of appropriate degree
            monoms = [exponent_to_polynomial(L, v) for L in sk_expo(deg - d, len(v))]
            result.extend(m*u for m in monoms)

    return vector_basis(result)

def vector_to_derivative(u: Union[List, vector]) -> Union[List, vector]:
    """Convert vector representation to partial derivatives.

    Args:
        u: Input vector or list of vectors

    Returns:
        Converted representation using partial derivatives
    """
    if isinstance(u, (list, tuple)):
        return [vector_to_derivative(elem) for elem in u]

    S = u.parent().base_ring()
    K = S.base_ring()
    variables = S.gens()

    # Create polynomial ring with both original and derivative variables
    PD = PolynomialRing(K, 'd', len(variables))
    FM = PolynomialRing(K, list(variables) + list(PD.gens()))
    derivatives = FM.gens()[len(variables):]

    return sum(u[j] * derivatives[j] for j in range(len(variables)))

def derivative_to_vector(u: Union[List, Any]) -> Union[List, vector]:
    """Convert partial derivatives to vector representation.

    Args:
        u: Input expression or list of expressions

    Returns:
        Vector representation
    """
    if isinstance(u, (list, tuple)):
        return [derivative_to_vector(elem) for elem in u]

    gens = u.parent().gens()
    mid = len(gens) // 2
    derivatives = gens[mid:]

    result = []
    for i, _ in enumerate(derivatives):
        substitutions = {d: 0 for d in derivatives}
        substitutions[derivatives[i]] = 1
        result.append(u.subs(substitutions))

    return vector(result)

def gendic(generators: List) -> Dict[int, List]:
    """Arrange generators with respect to degrees.

    Args:
        generators: List of generators

    Returns:
        Dictionary mapping degrees to lists of generators
    """
    result = {}
    for gen in generators:
        degree = max(m.degree() for m in gen)
        result.setdefault(degree, []).append(gen)
    return result

def module_intersection(modules: List[matrix]) -> matrix:
    """Compute the intersection of multiple modules.

    Args:
        modules: List of modules to intersect

    Returns:
        Intersection of the modules
    """
    if not modules:
        raise ValueError("Empty list of modules")

    result = modules[0].transpose()
    for mod in modules[1:]:
        result = singular.intersect(result, mod.transpose())

    return singular.minbase(result).sage().transpose()

# computation of D(A) by linear algebraic methods

def sk_expo(k, n):
    """
    Enumerate all exponents of monomials in S_k.

    INPUT:

    - ``k`` -- non-negative integer, the degree
    - ``n`` -- positive integer, number of variables

    OUTPUT:

    List of tuples representing exponents of monomials in S_k

    EXAMPLES::

        sage: sk_expo(2, 2)
        [(2, 0), (1, 1), (0, 2)]
    """
    Sk_list = []
    ones = np.ones(n, dtype=int)
    for e in Compositions(k + n, length=n): #TODO: better indexing?
        Sk_list.append(tuple(np.array(e) - ones))
    return Sk_list

def exponent_to_polynomial(Ls, v):
    """
    Convert a list of exponents to a polynomial in indeterminants v[].

    INPUT:

    - ``Ls`` -- list of tuples or single tuple of exponents
    - ``v`` -- list of variables

    OUTPUT:

    The polynomial corresponding to the given exponents

    EXAMPLES::

        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: exponent_to_polynomial([(2,0), (1,1)], [x,y])
        x^2 + x*y
    """
    if not isinstance(Ls, list):
        Ls = [Ls]
    result = sum(prod(v[i]**L[i] for i in range(len(L))) for L in Ls)
    return result

def coef_map(k, S):
    """
    Compute matrices representing multiplication by x[i]: S_k to S_{k+1}.

    This implementation is faster than coef_map2 and works for any characteristic.

    INPUT:

    - ``k`` -- non-negative integer, the degree
    - ``S`` -- polynomial ring

    OUTPUT:

    - List of matrices representing multiplication by each variable
    - List of basis elements of S_{k+1}

    EXAMPLES::

        sage: R = PolynomialRing(QQ, 'x', 2)
        sage: M, basis = coef_map(1, R)
        sage: len(M)  # number of variables
        2
    """
    v = S.gens()
    K = S.base_ring()
    n = len(v)

    # Get bases for S_k and S_{k+1}
    Sk_list = sk_expo(k, n)
    Sk1_list = sk_expo(k + 1, n)
    Sk1_dic = {ex: i for i, ex in enumerate(Sk1_list)}

    # Compute multiplication matrices
    Mi = [[] for i in range(n)]
    eye = np.eye(n)
    for i, expo in enumerate(Sk_list):
        for j in range(n):
            e = tuple(np.array(expo) + eye[j])
            Mi[j].append((i, Sk1_dic[e]))

    # Create sparse matrices
    M = [MatrixArgs(K, len(Sk_list), len(Sk1_list),
                   entries={k: 1 for k in Mi[j]}).matrix()
         for j in range(n)]

    return M, [exponent_to_polynomial(expo, v) for expo in Sk1_list]

def image_lambda(V, n):
    """
    Compute the image of lambda_i(multiplication by x[i]) of V.

    INPUT:

    - ``V`` -- list of vector representations of elements of S^n
    - ``n`` -- number of variables

    OUTPUT:

    Matrix representing the image

    EXAMPLES::

        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: V = [vector([x, y])]
        sage: M = image_lambda(V, 2)
        sage: M.nrows()
        6
    """
    # Find maximum degree
    k = max(V[0][j].degree() for j in range(n))

    # Set up basis for S_{k+1}
    Sk1_list = sk_expo(k + 1, n)
    Sk1_dic = {e: i for i, e in enumerate(Sk1_list)}
    eye = np.eye(n, dtype=int)

    # Compute entries of the matrix
    L = {}
    num = 0
    for u in V:
        for j in range(n):  # times v[j]
            for i in range(n):  # each exponent
                for e, c in u[i].dict().items():
                    le = tuple(np.array(e[:n]) + eye[j])
                    L[Sk1_dic[le]*n + i, num] = c
            num += 1

    return MatrixArgs(QQ, len(Sk1_dic)*n, len(V)*n, entries=L).matrix()

def basis_da(mat, k0):
    """
    Compute a linear basis of D(A) of degree k0.

    INPUT:

    - ``mat`` -- matrix representing the hyperplane arrangement
    - ``k0`` -- positive integer, the degree

    OUTPUT:

    List of vectors forming a basis of D(A) in degree k0

    EXAMPLES::

        sage: mat = matrix(QQ, [[1,0], [0,1]])
        sage: basis = basis_da(mat, 1)
        sage: len(basis)
        2
    """
    k = k0 - 1
    p = mat.nrows()
    n = mat.ncols()
    K = mat.base_ring()
    S = PolynomialRing(K, 'x', n)

    # Get multiplication matrices and basis
    M, Sk1 = coef_map(k, S)
    m1 = binomial(n + k - 1, k)
    m2 = len(Sk1)

    # Set up coefficient matrix
    C = zero_matrix(K, n*m2 + p*m1, p*m2)

    for i in range(p):
        D = zero_matrix(K, n*m2 + p*m1, m2)
        Y = sum(mat[i, j]*M[j] for j in range(n))
        D[(n*m2 + i*m1):(n*m2 + (i + 1)*m1), :] = Y
        for j in range(m2):
            D[j*n:(j + 1)*n, j] = mat[i, :].transpose()
        C[:, i*m2:(i + 1)*m2] = D

    # Compute kernel
    B = C.sparse_matrix().left_kernel().matrix()
    B = B[:, :n*m2]

    # Convert to vector field basis
    basis = []
    for i in range(B.nrows()):
        if B[i] != 0:
            E = sum(Sk1[j]*B[i, j*n:(j + 1)*n] for j in range(m2))
            basis.append(vector(E))

    return basis

def min_gen_arr_linear(mat, max_k=None, verbose=True):
    """
    Find minimal set of homogeneous generators by row echelon form.

    INPUT:

    - ``mat`` -- matrix representing the hyperplane arrangement
    - ``max_k`` -- (optional) maximum degree to consider
    - ``verbose`` -- (default: True) whether to print progress information

    OUTPUT:

    List of minimal generators

    EXAMPLES::

        sage: mat = matrix(QQ, [[1,0], [0,1]])
        sage: gens = min_gen_arr_linear(mat, max_k=2)
        sage: len(gens)
        2
    """
    p = mat.nrows()
    n = mat.ncols()
    max_k = p - n + 1 if max_k is None else max_k

    K = mat.base_ring()
    S = PolynomialRing(K, 'x', n)
    v = S.gens()

    # Get degree 1 basis
    V = basis_da(mat, 1)
    if verbose:
        print(f'deg 1, num gens {len(V)}')

    MG = V
    for k in range(max_k - 1):
        M = image_lambda(V, n)
        V = basis_da(mat, k + 2)

        # Set up basis for S_{k+1}
        Sk1_list = sk_expo(k + 2, n)
        Sk1_dic = {e: i for i, e in enumerate(Sk1_list)}
        Sk1 = [exponent_to_polynomial(expo, v) for expo in Sk1_list]

        # Compute new generators
        V1 = vector_to_flatten(V, Sk1_dic)
        lDk1 = M.augment(V1)
        piv = np.array(lDk1.pivots())
        mask = piv >= M.ncols()

        new_gens = [vector(sum(Sk1[j]*lDk1[j*n:(j + 1)*n, i]
                             for j in range(len(Sk1))))
                   for i in piv[mask]]
        MG.extend(new_gens)

        if verbose:
            print(f'deg {k + 2}, dim {len(V)}, '
                  f'codim {n*binomial(n + k, k + 1) - len(V)}, '
                  f'num new gen {sum(mask)}')

    return MG


def minimal_generating_set(G):
    """
    Find minimal set of homogeneous generators for a submodule of S^n.

    For homogeneous input, use minbase() which is more efficient.

    INPUT:

    - ``G`` -- list of module elements

    OUTPUT:

    List of minimal generators

    EXAMPLES::

        sage: R.<x,y> = PolynomialRing(QQ, 2)
        sage: G = [vector([x^2, 0]), vector([x^2, 0])]
        sage: min_gens = minimal_generating_set(G)
        sage: len(min_gens)
        1
    """
    if not G:
        return []

    # Set up polynomial rings
    S = G[0].parent().base_ring()
    v = S.gens()
    K = S.base_ring()
    PD = PolynomialRing(K, 'd', len(v))
    FM = PolynomialRing(K, list(v) + list(PD.gens()))
    d = FM.gens()[len(v):]

    # Sort generators by degree
    G = sorted(G, key=lambda x: x.degree())

    # Relations among partial derivatives
    r = [d[i]*d[j] for i in range(len(v))
         for j in range(i, len(v))]

    # Find minimal generators
    MG = []
    MG_index = []
    I = Ideal(*r, *MG)

    for i, g in enumerate(G):
        g_d = vector_to_derivative(g)
        if g_d not in I:
            MG.append(g_d)
            MG_index.append(i)
            I = Ideal(*r, *MG)

    return [G[i] for i in MG_index]

def min_gen_arr(mat, verbose=True):
    """
    Find minimal set of homogeneous generators for D(A).

    Uses Saito's criterion to optimize the computation.

    INPUT:

    - ``mat`` -- matrix representing the hyperplane arrangement
    - ``verbose`` -- (default: True) whether to print progress information

    OUTPUT:

    List of minimal generators for D(A)

    EXAMPLES::

        sage: mat = matrix(QQ, [[1,0], [0,1]])
        sage: gens = min_gen_arr(mat)
        sage: len(gens)
        2

    ALGORITHM:

    1. Check for duplicate planes
    2. Set up polynomial rings
    3. For each degree k:
       - Check Saito's criterion
       - Generate basis elements
       - Add independent generators
       - Check freeness condition
    """
    # Validate input
    if not is_distinct_planes(mat):
        raise ValueError('The arrangement contains duplicated planes!')

    p = mat.nrows()
    n = mat.ncols()
    K = mat.base_ring()

    # Set up polynomial rings
    S = PolynomialRing(K, 'x', n)
    v = S.gens()
    PD = PolynomialRing(K, 'd', len(v))
    FM = PolynomialRing(K, list(v) + list(PD.gens()))
    d = FM.gens()[len(v):]

    if verbose:
        print(f'Number of planes: {p}')

    # Relations among partial derivatives
    r = [d[i]*d[j] for i in range(len(v))
         for j in range(i, len(v))]

    GEN = []
    GEN_vec = []
    degs = []

    for k in range(p - n + 2):
        # Check Saito's criterion
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

        # Generate module basis
        G = basis_da(mat, k + 1)
        if verbose:
            print(f'Number of vector generators at deg {k + 1}: {len(G)}')

        # Find new independent generators
        I = Ideal(*r, *GEN)
        for g in G:
            g_d = vector_to_derivative(g)
            if g_d not in I:
                GEN.append(g_d)
                GEN_vec.append(g)
                I = Ideal(*r, *GEN)
                degs.append(k + 1)

                # Check freeness by Saito's criterion
                if len(degs) == len(v) and sum(degs) == p:
                    if verbose:
                        print('Arrangement is free')
                        print('Degree sequence:', degs)
                    return GEN_vec

    if verbose:
        print('Degree sequence:', degs)
    return GEN_vec

# Least squares fitting of polynomial vector field (v.f.) on a convex polygon
def dehomogenise(A, G):
    """
    Eliminate the last entry by subtracting Euler vector field.

    INPUT:

    - ``A`` -- hyperplane arrangement
    - ``G`` -- list of vector fields

    OUTPUT:

    List of dehomogenised vector fields

    """
    v1 = A.S.gens()
    return [A.euler_complement(g, A.n - 1).subs({v1[A.n - 1]: 1}) for g in G]


def affine_basis(A, G):
    r"""
    Return an affine-chart basis suitable for planar reconstruction.

    INPUT:

    - ``A`` -- hyperplane arrangement.
    - ``G`` -- list of generators in the homogeneous module, or an already
      affine list of vector fields with ``A.n - 1`` components.

    OUTPUT:

    Vector-space basis of affine vector fields on the chart ``x_n = 1``.
    For homogeneous inputs this drops the final component after
    dehomogenisation.

    """
    if not G:
        return []
    if len(G[0]) == A.n - 1:
        affine_fields = [vector(g) for g in G]
    else:
        affine_fields = [vector(g[:A.n - 1]) for g in dehomogenise(A, G)]

    maxdeg = degseq(affine_fields)[-1]
    if maxdeg < 1:
        return [affine_fields[0]]

    Sk_list = []
    for k in range(maxdeg + 1):
        Sk_list.extend(sk_expo(k, len(affine_fields[0].base_ring().gens())))
    Sk_dic = {e: i for i, e in enumerate(Sk_list)}
    if len(Sk_dic) == 0:
        return []

    V1 = vector_to_flatten(affine_fields, Sk_dic, n=A.n - 1)
    return [affine_fields[i] for i in V1.pivots()]

def fit_vf(A, Obs, mod_gens, verbose=True):
    """
    Fit a vector field to observations using least squares.

    INPUT:

    - ``A`` -- hyperplane arrangement
    - ``Obs`` -- dictionary mapping points to observed vectors
    - ``mod_gens`` -- module generators to use as basis
    - ``verbose`` -- (default: True) whether to print progress

    OUTPUT:

    - fitted vector field
    - residual error

    """
    v1 = A.S.gens()
    G = affine_basis(A, mod_gens)
    if verbose:
        print(f'Basis dimension: {len(G)}')

    # Normalize basis vector fields to prevent numerical overflow
    normalized_G = []
    for i, vf in enumerate(G):
        norm_factor = 1
        try:
            # Find max coefficient across all components
            max_coeff = 0
            for comp in vf:
                coeffs = comp.coefficients()
                if coeffs:
                    comp_max = max(abs(c) for c in coeffs)
                    max_coeff = max(max_coeff, comp_max)

            if max_coeff > 0:
                try:
                    max_coeff_float = float(max_coeff)
                    if not np.isfinite(max_coeff_float):
                        # Coefficient too large, normalize symbolically
                        vf = vector([comp / max_coeff for comp in vf])
                    else:
                        # Normalize by max coefficient
                        vf = vector([comp / max_coeff for comp in vf])
                except (OverflowError, ValueError):
                    # Coefficient overflow, normalize symbolically
                    vf = vector([comp / max_coeff for comp in vf])
        except Exception:
            pass
        normalized_G.append(vf)

    G = normalized_G

    # Prepare least squares system (robust evaluation over QQ then cast to float)
    X_rows, Y = [], []
    try:
        for p, u in Obs.items():
            # exact substitution dictionary in QQ when possible
            try:
                subs_q = {v1[i]: QQ(p[i]) for i in range(A.n - 1)}
            except Exception:
                # fallback to RR if QQ conversion fails
                subs_q = {v1[i]: RR(p[i]) for i in range(A.n - 1)}

            # Evaluate each basis vector field component-wise to avoid coercion issues
            # M is len(G) x (A.n-1)
            M = []
            for base_vf in G:
                comps = []
                for i in range(A.n - 1):
                    val = base_vf[i].subs(subs_q)
                    try:
                        fv = float(val)
                    except Exception:
                        fv = RR(val)
                        fv = float(fv)
                    comps.append(fv)
                M.append(comps)

            M = np.array(M, dtype=np.float64)  # shape: (#basis, dim)

            # Skip if any non-finite values
            if not np.all(np.isfinite(M)):
                continue

            # Build system rows for each component
            for i in range(A.n - 1):
                X_rows.append(M[:, i])
            Y.append(np.array(u, dtype=np.float64))

        if not X_rows:
            raise ValueError("No valid observations after evaluation; check filtering or data.")

        X = np.vstack(X_rows)
        Y = np.vstack(Y).ravel()

        # Solve least squares problem
        x, _, _, _ = lstsq(X, Y)
        res = float(np.sum((X.dot(x) - Y)**2))  # residual

        # Construct resulting vector field
        derivation = (matrix(G).transpose()) * vector(x.ravel())
        return derivation, res

    except Exception as e:
        raise ValueError(f"Error in fitting: {str(e)}")


def fit_vorticity(A, Obs, mod_gens, verbose=True):
    r"""
    Fit a vector field so that its vorticity matches sampled scalar data.

    INPUT:

    - ``A`` -- hyperplane arrangement corresponding to a 2D physical domain
      (i.e. ``A.n - 1 == 2``).
    - ``Obs`` -- dictionary mapping planar points ``(x, y)`` to observed
      vorticity values ``\omega_z``.
    - ``mod_gens`` -- list of module generators providing the fitting basis.
    - ``verbose`` -- (default: True) whether to print progress information.

    OUTPUT:

    Tuple ``(vector_field, residual)`` analogous to :func:`fit_vf` but obtained
    from vorticity-only measurements.

    """
    if A.n - 1 != 2:
        raise NotImplementedError('vorticity fitting currently supports only 2D domains')

    v1 = A.S.gens()
    basis = affine_basis(A, mod_gens)
    if verbose:
        print(f'Basis dimension: {len(basis)}')

    vort_basis = [rot(vf, A.S) for vf in basis]

    # Normalize each vorticity expression to prevent numerical overflow
    # Also normalize the basis in the same way for consistent scaling
    normalized_vort_basis = []
    normalized_basis = []
    norm_factors = []
    for i, vort_expr in enumerate(vort_basis):
        norm_factor = 1
        try:
            coeffs = vort_expr.coefficients()
            if coeffs:
                max_coeff = max(abs(c) for c in coeffs)
                if max_coeff > 0:
                    # Check if max_coeff is representable as a float
                    try:
                        max_coeff_float = float(max_coeff)
                        if not np.isfinite(max_coeff_float):
                            # Coefficient too large, normalize using symbolic division
                            vort_expr = vort_expr / max_coeff
                            norm_factor = max_coeff
                        else:
                            # Divide by max coefficient to normalize
                            vort_expr = vort_expr / max_coeff
                            norm_factor = max_coeff
                    except (OverflowError, ValueError):
                        # Coefficient overflow, still try to normalize symbolically
                        vort_expr = vort_expr / max_coeff
                        norm_factor = max_coeff
        except Exception as e:
            if verbose:
                print(f"Vorticity basis[{i}] normalization failed: {e}")
        normalized_vort_basis.append(vort_expr)
        # Normalize the corresponding basis vector field by the same factor
        normalized_basis.append(basis[i] / norm_factor if norm_factor != 1 else basis[i])
        norm_factors.append(norm_factor)

    vort_basis = normalized_vort_basis
    basis = normalized_basis

    rows = []
    rhs = []
    for point, omega in Obs.items():
        if len(point) != A.n - 1:
            raise ValueError('each observation key must match the spatial dimension of the arrangement')
        try:
            subs_q = {v1[i]: QQ(point[i]) for i in range(A.n - 1)}
        except Exception:
            subs_q = {v1[i]: RR(point[i]) for i in range(A.n - 1)}
        subs_q.setdefault(v1[A.n - 1], 1)

        row_vals: List[float] = []
        finite_row = True
        for vort_expr in vort_basis:
            val = vort_expr.subs(subs_q)
            try:
                num = float(val)
            except Exception:
                try:
                    num = float(RR(val))
                except Exception:
                    finite_row = False
                    break
            if not np.isfinite(num):
                finite_row = False
                break
            row_vals.append(num)

        if not finite_row:
            continue

        rows.append(np.array(row_vals, dtype=np.float64))
        rhs.append(float(omega))

    if not rows:
        raise ValueError('no valid vorticity observations for fitting')

    X = np.vstack(rows)
    y = np.array(rhs, dtype=np.float64)

    coeffs, _, _, _ = lstsq(X, y)
    residual = float(np.sum((X.dot(coeffs) - y) ** 2))

    derivation = (matrix(basis).transpose()) * vector(coeffs)
    return derivation, residual

def given_min_error(A, P, e0, verbose=True):
    """
    Find minimum degree vector field achieving given error bound.

    INPUT:

    - ``A`` -- hyperplane arrangement
    - ``P`` -- observations
    - ``e0`` -- error bound
    - ``verbose`` -- (default: True) whether to print progress

    OUTPUT:

    - vector field
    - achieved error
    - minimum degree
    """
    G = A.minimal_generators()[1:]  # exclude Euler
    k = G[0][0].degree()
    if verbose:
        print(f'Starting with degree {k}')

    while True:
        mod_gens = graded_component(G, k)
        u, err = fit_vf(A, P, mod_gens, verbose=verbose)
        if err <= e0:
            return u, err, k
        k += 1
        if verbose:
            print(f'Trying degree {k}')

def div(u, S):
    """
    Compute divergence of vector field u.

    INPUT:

    - ``u`` -- vector field
    - ``S`` -- polynomial ring

    OUTPUT:

    Divergence ∇·u
    """
    v = S.gens()
    n_spatial = min(len(u), len(v))
    return sum(diff(u[i], v[i]) for i in range(n_spatial))

def rot(u, S):
    """
    Compute rotation of 2D vector field u.

    INPUT:

    - ``u`` -- vector field
    - ``S`` -- polynomial ring

    OUTPUT:

    Rotation ∇×u (scalar in 2D)

    NOTE: Uses the first two coordinates/variables of ``u`` and ``S``.
    """
    if len(u) < 2 or len(S.gens()) < 2:
        raise NotImplementedError("Rotation requires at least two coordinates")
    v = S.gens()
    return diff(u[1], v[0]) - diff(u[0], v[1])

def laplacian(u, S):
    """
    Compute Laplacian of vector field u.

    INPUT:

    - ``u`` -- vector field
    - ``S`` -- polynomial ring

    OUTPUT:

    Vector Laplacian ∇²u
    """
    v = S.gens()
    n_spatial = min(len(u), len(v))
    return [sum(diff(diff(u[j], v[i]), v[i]) for i in range(n_spatial))
            for j in range(len(u))]

def divergence_free(G):
    """
    Find divergence-free vector fields in span of G.

    INPUT:

    - ``G`` -- list of vector fields

    OUTPUT:

    Basis of divergence-free subspace
    """
    return _differential_free(G, div, 1)

def rotation_free(G):
    """
    Find rotation-free vector fields in span of G.

    INPUT:

    - ``G`` -- list of vector fields

    OUTPUT:

    Basis of rotation-free subspace
    """
    return _differential_free(G, rot, 1)

def harmonic(G):
    """
    Find harmonic vector fields in span of G.

    INPUT:

    - ``G`` -- list of vector fields

    OUTPUT:

    Basis of harmonic subspace
    """
    n = len(G[0].base_ring().gens())
    return _differential_free(G, laplacian, n - 1)

def _differential_free(G, op, n_comp):
    """
    Helper function for finding vector fields in the kernel of op.

    INPUT:

    - ``G`` -- list of vector fields
    - ``op`` -- operator to apply
    - ``n_comp`` -- number of components in operator output

    OUTPUT:

    Basis of kernel subspace
    """
    if not G:
        return []

    # Apply operator to basis
    S = G[0].parent().base_ring() # get the polynomial ring
    C = [op(u, S) for u in G]

    # Set up coefficient matrix
    n = len(S.gens())
    if n_comp == 1:
        maxdeg = max(g.degree() for g in C)
    else:
        maxdeg = max(max(m.degree() for m in g) for g in C)

    Sk_list = []
    for k in range(maxdeg + 1):
        Sk_list.extend(sk_expo(k, n))
    Sk_dic = {e: i for i, e in enumerate(Sk_list)}

    # Convert to matrix equation
    MD = vector_to_flatten(C, Sk_dic, n=n_comp)

    # Solve homogeneous system
    ker = MD.right_kernel().matrix()
    return [sum(a[i]*G[i] for i in range(len(G))) for a in ker]


HyperPlaneArr = HyperplaneArrangement

__all__ = [
    'HyperplaneArrangement',
    'HyperPlaneArr',
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
