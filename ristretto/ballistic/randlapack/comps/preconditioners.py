"""
Notes: these functions aren't currently used in least squares drivers.
It's not clear what the API should be.
Maybe sketch_and_factor needs a more general "method" argument, so it
can return QR, SVD, or something else.
"""
import scipy as sp
from scipy.linalg import lapack
from ristretto.ballistic.rblas import sketching as sk


def sketch_and_factor(S, A, compute_Q=True):
    # Letting the user skip Q is useful when solving positive definite systems that
    # don't arise as part of least-squares problems. There's also a potential performance
    # benefit to skipping computation of Q if we're happy initializing CG/LSQR at the
    # origin in sketch-and-precondition.
    Aske = S @ A
    if compute_Q:
        Q, R = sp.linalg.qr(Aske, mode='economic', pivoting=False)
    else:
        R = sp.linalg.qr(Aske, mode='r', pivoting=False)
        Q = None
    return R, Q


def gaussian_precond(A, d, compute_Q=True):
    n, m = A.shape
    assert n > d > m
    S = sk.gaussian_operator(n_rows=d, n_cols=n, normalize=True)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q


def srct_precond(A, d, compute_Q=True):
    n, m = A.shape
    assert n > d > m
    S = sk.srct_operator(n_rows=d, n_cols=n)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q


def iid_sparse_precond(A, d, density, compute_Q=True):
    assert density > 0
    assert density <= 1
    n, m = A.shape
    assert n > d > m
    S = sk.sparse_sign_operator(n_rows=d, n_cols=n, density=density)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q


def fixed_sparse_precond(A, d, col_nnz, compute_Q=True):
    n, m = A.shape
    assert n > d > m
    S = sk.sjlt_operator(n_rows=d, n_cols=n, vec_nnz=col_nnz)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q
