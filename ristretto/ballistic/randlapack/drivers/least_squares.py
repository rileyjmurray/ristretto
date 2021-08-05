"""
Routines for (approximately) solving over-determined least squares problems

    min{ || A x - b ||_2 + reg || x ||_2 : x in R^n }.
"""
import scipy.linalg as la
import numpy as np
from ristretto.ballistic.randlapack.comps import deterministic as de


def sketch_and_solve(A, b, S, cond=None, check_finite=True, lapack_driver=None):
    Aske = S @ A
    bske = S @ b
    res = la.lstsq(Aske, bske, cond, True, True, check_finite, lapack_driver)
    # ^ Overwrite Aske and bske.
    return res


def sketch_and_precond(A, b, S, tol, iter_lim, method='qr'):
    if method == 'qr':
        Aske = S @ A
        Q, R = la.qr(Aske, overwrite_a=True, mode='economic')
        b_ske = S @ b
        x_ske = la.solve_triangular(R, Q.T @ b_ske, lower=False)
        x0 = None
        if np.linalg.norm(A @ x_ske - b) < np.linalg.norm(b):
            x0 = x_ske
        res = de.upper_tri_precond_lsqr(A, b, R, tol, iter_lim, x0=x0)
        return res
    elif method == 'svd':
        Aske = S @ A
        U, sigma, Vh = la.svd(Aske, overwrite_a=True, check_finite=False, full_matrices=False)
        rank = np.count_nonzero(sigma > sigma[0] * np.min(A.shape) * np.finfo(float).eps)
        N = Vh[:rank, :].T / sigma[:rank]
        #TODO: compute the solution to (Aske' Aske) x_ske = Aske' bske using the SVD, then update
        #   b = b - A x_ske, then run preconditioned LSQR to get x, and finally
        #   update x += x_ske.
        res = de.pinv_precond_lsqr(A, b, N, tol, iter_lim)
        return res
    else:
        raise ValueError()
