"""
A library of well-known hyperplane arrangements.
"""

from sage.all import matrix, QQ
from .arrangement import HyperplaneArrangement


def braid(n, base_field=QQ):
    """
    Return the braid arrangement (Coxeter arrangement of type A_{n-1}) in R^n.
    The hyperplanes are x_i - x_j = 0 for 1 <= i < j <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row = [base_field(0)] * n
            row[i] = base_field(1)
            row[j] = base_field(-1)
            rows.append(row)
    if not rows:
        return HyperplaneArrangement(matrix(base_field, 0, n))
    return HyperplaneArrangement(matrix(base_field, rows))


def type_B(n, base_field=QQ):
    """
    Return the Coxeter arrangement of type B_n (also C_n) in R^n.
    The hyperplanes are x_i +/- x_j = 0 for 1 <= i < j <= n, and x_i = 0 for 1 <= i <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row1 = [base_field(0)] * n
            row2 = [base_field(0)] * n
            row1[i] = base_field(1)
            row1[j] = base_field(-1)
            row2[i] = base_field(1)
            row2[j] = base_field(1)
            rows.append(row1)
            rows.append(row2)
        row3 = [base_field(0)] * n
        row3[i] = base_field(1)
        rows.append(row3)
    if not rows:
        return HyperplaneArrangement(matrix(base_field, 0, n))
    return HyperplaneArrangement(matrix(base_field, rows))


def type_D(n, base_field=QQ):
    """
    Return the Coxeter arrangement of type D_n in R^n.
    The hyperplanes are x_i +/- x_j = 0 for 1 <= i < j <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            row1 = [base_field(0)] * n
            row2 = [base_field(0)] * n
            row1[i] = base_field(1)
            row1[j] = base_field(-1)
            row2[i] = base_field(1)
            row2[j] = base_field(1)
            rows.append(row1)
            rows.append(row2)
    if not rows:
        return HyperplaneArrangement(matrix(base_field, 0, n))
    return HyperplaneArrangement(matrix(base_field, rows))


def shi(n, base_field=QQ):
    """
    Return the cone of the Shi arrangement of type A_{n-1} in R^{n+1}.
    The affine hyperplanes are x_i - x_j = 0, 1 for 1 <= i < j <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    normals = []
    offsets = []
    for i in range(n):
        for j in range(i + 1, n):
            normal = [0] * n
            normal[i] = 1
            normal[j] = -1
            normals.append(tuple(normal))
            offsets.append([0, 1])
    if not normals:
        return HyperplaneArrangement(matrix(base_field, [[1] + [0]*n]))
    return HyperplaneArrangement.cone_of_arrangement(normals, offsets, base_field=base_field)


def catalan(n, base_field=QQ):
    """
    Return the cone of the Catalan arrangement of type A_{n-1} in R^{n+1}.
    The affine hyperplanes are x_i - x_j = -1, 0, 1 for 1 <= i < j <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    normals = []
    offsets = []
    for i in range(n):
        for j in range(i + 1, n):
            normal = [0] * n
            normal[i] = 1
            normal[j] = -1
            normals.append(tuple(normal))
            offsets.append([-1, 0, 1])
    if not normals:
        return HyperplaneArrangement(matrix(base_field, [[1] + [0]*n]))
    return HyperplaneArrangement.cone_of_arrangement(normals, offsets, base_field=base_field)


def linial(n, base_field=QQ):
    """
    Return the cone of the Linial arrangement of type A_{n-1} in R^{n+1}.
    The affine hyperplanes are x_i - x_j = 1 for 1 <= i < j <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    normals = []
    offsets = []
    for i in range(n):
        for j in range(i + 1, n):
            normal = [0] * n
            normal[i] = 1
            normal[j] = -1
            normals.append(tuple(normal))
            offsets.append([1])
    if not normals:
        return HyperplaneArrangement(matrix(base_field, [[1] + [0]*n]))
    return HyperplaneArrangement.cone_of_arrangement(normals, offsets, base_field=base_field)


def ish(n, base_field=QQ):
    """
    Return the cone of the Ish arrangement in R^{n+1}.
    The affine hyperplanes are x_i - x_j = 0 for 1 <= i < j <= n,
    and x_1 - x_j = i for 1 <= i < j <= n.
    """
    if n <= 0:
        return HyperplaneArrangement(matrix(base_field, 0, 0))
    normals = []
    offsets = []
    for i in range(n):
        for j in range(i + 1, n):
            normal = [0] * n
            normal[i] = 1
            normal[j] = -1
            normals.append(tuple(normal))
            if i == 0:
                offsets.append(list(range(j + 1)))
            else:
                offsets.append([0])
    if not normals:
        return HyperplaneArrangement(matrix(base_field, [[1] + [0]*n]))
    return HyperplaneArrangement.cone_of_arrangement(normals, offsets, base_field=base_field)

def exceptional_coxeter(type_name, base_field=QQ):
    """
    Return the exceptional Coxeter arrangement of the given type.
    Supported types: 'E6', 'E7', 'E8', 'F4', 'G2', 'H3', 'H4'.
    
    For E, F, G types, the ambient space representation of the root system is used.
    For H3 and H4, coordinates involving the golden ratio are used, and the
    base field is extended if necessary (to QQbar).
    """
    type_name = type_name.upper()
    
    if type_name in ['E6', 'E7', 'E8', 'F4', 'G2']:
        from sage.all import RootSystem
        rs = RootSystem(type_name).ambient_space()
        roots = list(rs.positive_roots())
        rows = [list(r.to_vector()) for r in roots]
        return HyperplaneArrangement(matrix(base_field, rows))
        
    elif type_name in ['H3', 'H4']:
        from sage.all import NumberField, PolynomialRing, QQ, RLF
        import itertools
        
        # Golden ratio via NumberField for performance
        K = NumberField(PolynomialRing(QQ, 'x')([ -1, -1, 1 ]), 'tau', embedding=RLF(1.618034))
        tau = K.gen()
        inv_tau = 1 / tau
        
        if type_name == 'H3':
            all_roots = []
            for i in range(3):
                r = [0, 0, 0]
                r[i] = 1
                all_roots.append(r)
                
            base_perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            for p in base_perms:
                for signs in itertools.product([1, -1], repeat=3):
                    val = [1, tau, inv_tau]
                    r = [0, 0, 0]
                    for i in range(3):
                        r[p[i]] = signs[i] * val[i]
                    all_roots.append(r)
                    
            pos_roots = []
            for r in all_roots:
                for x in r:
                    if x != 0:
                        if float(x) > 0:
                            pos_roots.append(r)
                        break
            
            return HyperplaneArrangement(matrix(K, pos_roots))
            
        elif type_name == 'H4':
            all_roots = []
            for i in range(4):
                r = [0]*4; r[i] = 1; all_roots.append(r)
                
            for signs in itertools.product([1, -1], repeat=4):
                all_roots.append([s for s in signs])
                
            def is_even(p):
                r"""
                Check if the value is even.
                """
                inversions = sum(1 for i in range(4) for j in range(i+1, 4) if p[i] > p[j])
                return inversions % 2 == 0
                
            even_perms = [p for p in itertools.permutations([0,1,2,3]) if is_even(p)]
            
            for p in even_perms:
                for signs in itertools.product([1, -1], repeat=3):
                    val = [0, 1, tau, inv_tau]
                    r = [0, 0, 0, 0]
                    r[p[0]] = 0
                    r[p[1]] = signs[0] * val[1]
                    r[p[2]] = signs[1] * val[2]
                    r[p[3]] = signs[2] * val[3]
                    all_roots.append(r)
                    
            pos_roots = []
            for r in all_roots:
                for x in r:
                    if x != 0:
                        if float(x) > 0:
                            pos_roots.append(r)
                        break
                        
            return HyperplaneArrangement(matrix(K, pos_roots))
            
    else:
        raise ValueError(f"Unknown exceptional Coxeter type: {type_name}")
