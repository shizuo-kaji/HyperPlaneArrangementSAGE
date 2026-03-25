import numpy as np
from sage.structure.sage_object import SageObject
from sage.modules.free_module_element import vector
from sage.matrix.constructor import matrix
from sage.rings.polynomial.polynomial_ring_constructor import PolynomialRing
from sage.rings.ideal import Ideal
from sage.matrix.args import MatrixArgs
from sage.rings.rational_field import QQ
from sage.all import diff
from .utils import sk_expo, exponent_to_polynomial

class VectorField(SageObject):
    """
    A class representing a polynomial vector field.
    """
    def __init__(self, u, S=None):
        from sage.structure.element import Vector
        self._u = u if isinstance(u, Vector) else vector(u)
        self.S = S if S is not None else self._u.parent().base_ring()

    @property
    def v(self):
        return self._u

    def __len__(self):
        return len(self._u)

    def __getitem__(self, item):
        return self._u[item]

    def __iter__(self):
        return iter(self._u)

    def __getattr__(self, name):
        """Delegate missing attributes to underlying vector."""
        return getattr(self._u, name)

    def __add__(self, other):
        if isinstance(other, VectorField):
            return VectorField(self._u + other._u, self.S)
        return VectorField(self._u + other, self.S)

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        if isinstance(other, VectorField):
            return VectorField(self._u - other._u, self.S)
        return VectorField(self._u - other, self.S)

    def __mul__(self, other):
        # Scalar multiplication
        return VectorField(self._u * other, self.S)

    def __rmul__(self, other):
        return VectorField(other * self._u, self.S)

    def __truediv__(self, other):
        return VectorField(self._u / other, self.S)
        
    def _repr_(self):
        return repr(self._u)

    def _latex_(self):
        from sage.misc.latex import latex
        return latex(self._u)

    def degree(self):
        return max(m.degree() for m in self._u)

    def div(self):
        """Compute divergence of vector field."""
        v = self.S.gens()
        n_spatial = min(len(self._u), len(v))
        return sum(diff(self._u[i], v[i]) for i in range(n_spatial))

    def rot(self):
        """Compute rotation of 2D vector field."""
        v = self.S.gens()
        if len(self._u) < 2 or len(v) < 2:
            raise NotImplementedError("Rotation requires at least two coordinates")
        return diff(self._u[1], v[0]) - diff(self._u[0], v[1])

    def laplacian(self):
        """Compute Laplacian of vector field (∇²u)."""
        v = self.S.gens()
        n_spatial = min(len(self._u), len(v))
        data = [sum(diff(diff(self._u[j], v[i]), v[i]) for i in range(n_spatial))
                for j in range(len(self._u))]
        return VectorField(data, self.S)

    def to_derivative(self):
        """Convert vector representation to partial derivatives."""
        K = self.S.base_ring()
        variables = self.S.gens()
        
        PD = PolynomialRing(K, 'd', len(variables))
        FM = PolynomialRing(K, list(variables) + list(PD.gens()))
        derivatives = FM.gens()[len(variables):]
        
        return sum(self._u[j] * derivatives[j] for j in range(len(variables)))

    @classmethod
    def from_derivative(cls, expr):
        """Convert partial derivatives to vector representation."""
        gens = expr.parent().gens()
        mid = len(gens) // 2
        derivatives = gens[mid:]
        
        result = []
        for i, _ in enumerate(derivatives):
            substitutions = {d: 0 for d in derivatives}
            substitutions[derivatives[i]] = 1
            result.append(expr.subs(substitutions))
            
        return cls(result)


class VectorFieldModule(SageObject):
    """
    A class representing a module of vector fields.
    """
    def __init__(self, generators):
        self.gens = [g if isinstance(g, VectorField) else VectorField(g) for g in generators]

    def __iter__(self):
        return iter(self.gens)

    def __len__(self):
        return len(self.gens)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return VectorFieldModule(self.gens[item])
        return self.gens[item]
        
    def _repr_(self):
        return f"Module of Vector Fields with {len(self.gens)} generators"

    def degs(self):
        """Return the sorted degree sequence of the derivations."""
        return sorted([g.degree() for g in self.gens])

    def is_s_indep(self):
        """Check S-linear independence via Saito's criterion."""
        return self.saito().det() != 0

    def saito(self):
        """Return the matrix used in Saito's criterion for freeness."""
        assert len(self.gens) > 0 and len(self.gens[0]) == len(self.gens)
        M = matrix([g.v for g in self.gens])
        return M

    def vector_basis(self):
        """Extract a vector-space basis from the generators."""
        if not self.gens:
            return VectorFieldModule([])
        n_vars = len(self.gens[0].v.base_ring().gens())
        n_comp = len(self.gens[0])
        maxdeg = self.degs()[-1]
        if maxdeg < 1:
            return VectorFieldModule([self.gens[0]])
        Sk_list = []
        for k in range(maxdeg + 1):
            Sk_list.extend(sk_expo(k, n_vars))
        Sk_dic = {e: i for i, e in enumerate(Sk_list)}
        if len(Sk_dic) == 0:
            return VectorFieldModule([])
        V1 = self.to_flatten(Sk_dic, n=n_comp)
        return VectorFieldModule([self.gens[i] for i in V1.pivots()])

    def to_flatten(self, Sk_dic, n=0):
        """Convert elements to flattened vectors based on degrees."""
        if not self.gens:
            raise ValueError("Input list cannot be empty")
        
        ngens = len(next(iter(Sk_dic)))
        nn = ngens if n == 0 else n
        
        L = {}
        for j, u in enumerate(self.gens):
            for i in range(nn):
                uu = u[i] if nn > 1 else u
                try:
                    for e, c in uu.dict().items():
                        L[Sk_dic[tuple(e[:ngens])]*nn + i, j] = c
                except (KeyError, AttributeError):
                    pass
                    
        return MatrixArgs(QQ, len(Sk_dic)*nn, len(self.gens), entries=L).matrix()

    def graded_component(self, deg):
        """Compute the graded component of target degree."""
        if not self.gens:
            return VectorFieldModule([])
            
        S = self.gens[0].S
        v = S.gens()
        
        result = []
        for d, u in zip(self.degs(), self.gens):
            if d <= deg:
                monoms = [exponent_to_polynomial(L, v) for L in sk_expo(deg - d, len(v))]
                result.extend(m * u for m in monoms)
                
        mod = VectorFieldModule(result)
        return mod.vector_basis()

    def gendic(self):
        """Arrange generators with respect to degrees."""
        result = {}
        for gen in self.gens:
            result.setdefault(gen.degree(), []).append(gen)
        return result

    def minimal_generating_set(self):
        """Find minimal set of homogeneous generators."""
        if not self.gens:
            return VectorFieldModule([])
            
        S = self.gens[0].S
        v = S.gens()
        K = S.base_ring()
        PD = PolynomialRing(K, 'd', len(v))
        FM = PolynomialRing(K, list(v) + list(PD.gens()))
        d = FM.gens()[len(v):]
        
        sorted_gens = sorted(self.gens, key=lambda x: x.degree())
        
        r = [d[i]*d[j] for i in range(len(v)) for j in range(i, len(v))]
        
        MG = []
        MG_index = []
        I = Ideal(*r, *MG)
        
        for i, g in enumerate(sorted_gens):
            g_d = g.to_derivative()
            if g_d not in I:
                MG.append(g_d)
                MG_index.append(i)
                I = Ideal(*r, *MG)
                
        return VectorFieldModule([sorted_gens[i] for i in MG_index])

    def image_lambda(self, n):
        """Compute the image of lambda_i (multiplication by x[i])."""
        if not self.gens:
            return matrix(QQ, 0, 0)
        
        k = max(g.degree() for g in self.gens)
        Sk1_list = sk_expo(k + 1, n)
        Sk1_dic = {e: i for i, e in enumerate(Sk1_list)}
        eye = np.eye(n, dtype=int)
        
        L = {}
        num = 0
        for u in self.gens:
            for j in range(n):
                for i in range(n):
                    for e, c in u[i].dict().items():
                        le = tuple(np.array(e[:n]) + eye[j])
                        L[Sk1_dic[le]*n + i, num] = c
                num += 1
                
        return MatrixArgs(QQ, len(Sk1_dic)*n, len(self.gens)*n, entries=L).matrix()

    def _differential_free(self, op, n_comp):
        if not self.gens:
            return VectorFieldModule([])

        C = [op(u) for u in self.gens]
        
        S = self.gens[0].S
        n = len(S.gens())
        if n_comp == 1:
            maxdeg = max(c.degree() if hasattr(c, 'degree') else 0 for c in C)
        else:
            maxdeg = max(c.degree() for c in C)
            
        Sk_list = []
        for k in range(maxdeg + 1):
            Sk_list.extend(sk_expo(k, n))
        Sk_dic = {e: i for i, e in enumerate(Sk_list)}
        
        MD = self._flatten_scalars(C, Sk_dic, n_comp) if n_comp == 1 else VectorFieldModule(C).to_flatten(Sk_dic, n=n_comp)
        
        ker = MD.right_kernel().matrix()
        return VectorFieldModule([sum(a[i]*self.gens[i] for i in range(len(self.gens))) for a in ker])
        
    def _flatten_scalars(self, C, Sk_dic, n_comp):
        ngens = len(next(iter(Sk_dic)))
        L = {}
        for j, u in enumerate(C):
            try:
                for e, c in u.dict().items():
                    L[Sk_dic[tuple(e[:ngens])], j] = c
            except (KeyError, AttributeError):
                pass
        return MatrixArgs(QQ, len(Sk_dic), len(C), entries=L).matrix()

    def divergence_free(self):
        return self._differential_free(lambda u: u.div(), 1)

    def rotation_free(self):
        return self._differential_free(lambda u: u.rot(), 1)

    def harmonic(self):
        n = len(self.gens[0].S.gens())
        return self._differential_free(lambda u: u.laplacian(), n - 1)

    def dehomogenise(self):
        """
        Eliminate the last entry by subtracting Euler vector field.
        """
        if not self.gens:
            return VectorFieldModule([])
            
        S = self.gens[0].S
        n = len(S.gens())
        v1 = S.gens()
        from sage.modules.free_module_element import vector
        euler_v = vector(v1)
        
        dehom_gens = []
        for g in self.gens:
            gv = g.v
            comp = gv - (gv[n - 1] // v1[n - 1]) * euler_v
            ans_sub = comp.subs({v1[n - 1]: 1})
            dehom_gens.append(VectorField(ans_sub, S))
            
        return VectorFieldModule(dehom_gens)

    def affine_basis(self):
        """
        Return an affine-chart basis suitable for planar reconstruction.
        """
        if not self.gens:
            return VectorFieldModule([])
            
        S = self.gens[0].S
        n = len(S.gens())
        
        from sage.modules.free_module_element import vector
        if len(self.gens[0]) == n - 1:
            affine_fields = self
        else:
            affine_fields = VectorFieldModule([vector(g.v[:n - 1]) for g in self.dehomogenise().gens])

        return affine_fields.vector_basis()
