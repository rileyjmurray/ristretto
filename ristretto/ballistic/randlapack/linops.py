"""
Working with LinearOperator objects
"""
import scipy.sparse.linalg as sparla
import numpy as np
from ristretto.ballistic.rblas.sketching import apply_srct, \
    gaussian_operator, sjlt_operator, sparse_sign_operator, srct_data
# ^ Import some functions which might not be used in this file, but
#   which should be importable from this file.


def normalized_gaussian_operator(n_rows, n_cols, rng):
    return gaussian_operator(n_rows, n_cols, rng, normalize=True)


def srct_operator(n_rows, n_cols, rng=None):
    """
    Construct data for an SRTT based on the discrete cosine transform.
    Then, construct and return a representation for that SRTT as a SciPy
    LinearOperator. That LinearOperator uses rblas.sketching.apply_srct(...)
    as its implementation.
    """
    r, e, perm = srct_data(n_rows, n_cols, rng)

    if n_cols >= n_rows:
        def srct(mat):
            return apply_srct(r, e, mat, perm)

        S = sparla.LinearOperator(shape=(n_rows, n_cols),
                                  matvec=srct, matmat=srct)
        S.__dict__['sketch_data'] = (r, e, perm)
    else:
        def srct(mat):
            return apply_srct(r, e, mat.T, perm).T

        S = sparla.LinearOperator(shape=(n_rows, n_cols),
                                  rmatvec=srct, rmatmat=srct)
        S.__dict__['sketch_data'] = (r, e, perm)
    return S
