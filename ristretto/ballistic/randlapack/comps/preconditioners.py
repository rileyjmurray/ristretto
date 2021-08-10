"""
Notes: these functions aren't currently used in least squares drivers.
It's not clear what the API should be.
Maybe sketch_and_factor needs a more general "method" argument, so it
can return QR, SVD, or something else.
"""
import scipy as sp
import numpy as np
from scipy.linalg import lapack
from ristretto.ballistic.rblas import sketching as sk


def sketch_and_factor(S, A, compute_Q=True):
    # Letting the user skip Q is useful when solving positive definite systems that
    # don't arise as part of least-squares problems. There's also a potential performance
    # benefit to skipping computation of Q if we're happy initializing CG/LSQR at the
    # origin in sketch-and-precondition.
    A_ske = S @ A
    if compute_Q:
        Q, R = sp.linalg.qr(A_ske, mode='economic', pivoting=False)
    else:
        R = sp.linalg.qr(A_ske, mode='r', pivoting=False)
        Q = None
    return R, Q


def gaussian_precond(A, d, gen=None, compute_Q=True):
    n, m = A.shape
    assert n > d > m
    gen = np.random.default_rng(gen)
    S = sk.gaussian_operator(n_rows=d, n_cols=n, gen=gen, normalize=True)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q


def srct_precond(A, d, gen=None, compute_Q=True):
    n, m = A.shape
    assert n > d > m
    gen = np.random.default_rng(gen)
    S = sk.srct_operator(n_rows=d, n_cols=n, gen=gen)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q


def iid_sparse_precond(A, d, gen=None, compute_Q=True, density=0.05):
    assert density > 0
    assert density <= 1
    n, m = A.shape
    assert n > d > m
    gen = np.random.default_rng(gen)
    S = sk.sparse_sign_operator(n_rows=d, n_cols=n, gen=gen, density=density)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q


def fixed_sparse_precond(A, d, gen=None, compute_Q=True, col_nnz=8):
    n, m = A.shape
    assert n > d > m
    gen = np.random.default_rng(gen)
    S = sk.sjlt_operator(n_rows=d, n_cols=n, gen=gen, vec_nnz=col_nnz)
    R, Q = sketch_and_factor(S, A, compute_Q)
    return R, Q
