"""Fitting vector fields to data."""
import numpy as np
from scipy.linalg import lstsq
from sage.matrix.constructor import matrix
from sage.modules.free_module_element import vector
from sage.rings.rational_field import QQ
from sage.rings.real_mpfr import RR
from .vector_field import VectorField, VectorFieldModule

def fit_vf(A, Obs, mod_gens, verbose=True):
    v1 = A.S.gens()
    if isinstance(mod_gens, VectorFieldModule):
        basis = mod_gens.affine_basis()
    else:
        basis = VectorFieldModule(mod_gens).affine_basis()
        
    G = basis.gens
    if verbose:
        print(f'Basis dimension: {len(G)}')

    normalized_G = []
    for i, vf in enumerate(G):
        norm_factor = 1
        try:
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
                        vf = VectorField([comp / max_coeff for comp in vf])
                    else:
                        vf = VectorField([comp / max_coeff for comp in vf])
                except (OverflowError, ValueError):
                    vf = VectorField([comp / max_coeff for comp in vf])
        except Exception:
            pass
        normalized_G.append(vf)

    G = normalized_G

    X_rows, Y = [], []
    try:
        for p, u in Obs.items():
            try:
                subs_q = {v1[i]: QQ(p[i]) for i in range(A.n - 1)}
            except Exception:
                subs_q = {v1[i]: RR(p[i]) for i in range(A.n - 1)}

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

            M = np.array(M, dtype=np.float64)

            if not np.all(np.isfinite(M)):
                continue

            for i in range(A.n - 1):
                X_rows.append(M[:, i])
            Y.append(np.array(u, dtype=np.float64))

        if not X_rows:
            raise ValueError("No valid observations after evaluation; check filtering or data.")

        X = np.vstack(X_rows)
        Y = np.vstack(Y).ravel()

        x, _, _, _ = lstsq(X, Y)
        res = float(np.sum((X.dot(x) - Y)**2))

        derivation = (matrix([g.v for g in G]).transpose()) * vector(x.ravel())
        return VectorField(derivation), res

    except Exception as e:
        raise ValueError(f"Error in fitting: {str(e)}")

def fit_vorticity(A, Obs, mod_gens, verbose=True):
    if A.n - 1 != 2:
        raise NotImplementedError('vorticity fitting currently supports only 2D domains')

    v1 = A.S.gens()
    if isinstance(mod_gens, VectorFieldModule):
        basis = mod_gens.affine_basis().gens
    else:
        basis = VectorFieldModule(mod_gens).affine_basis().gens
        
    if verbose:
        print(f'Basis dimension: {len(basis)}')

    vort_basis = [vf.rot() for vf in basis]

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
                    try:
                        max_coeff_float = float(max_coeff)
                        if not np.isfinite(max_coeff_float):
                            vort_expr = vort_expr / max_coeff
                            norm_factor = max_coeff
                        else:
                            vort_expr = vort_expr / max_coeff
                            norm_factor = max_coeff
                    except (OverflowError, ValueError):
                        vort_expr = vort_expr / max_coeff
                        norm_factor = max_coeff
        except Exception as e:
            if verbose:
                print(f"Vorticity basis[{i}] normalization failed: {e}")
        normalized_vort_basis.append(vort_expr)
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

        row_vals = []
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

    derivation = (matrix([g.v for g in basis]).transpose()) * vector(coeffs)
    return VectorField(derivation), residual


def given_min_error(A, P, e0, verbose=True):
    G = VectorFieldModule(A.minimal_generators().gens[1:])  # exclude Euler
    k = G.gens[0].degree()
    if verbose:
        print(f'Starting with degree {k}')

    while True:
        mod_gens = G.graded_component(k).gens
        u, err = fit_vf(A, P, mod_gens, verbose=verbose)
        if err <= e0:
            return u, err, k
        k += 1
        if verbose:
            print(f'Trying degree {k}')
