import numpy as np
import scipy.sparse as spar
from scipy.fft import dct


def gaussian_operator(n_rows, n_cols, rng=None, normalize=True):
    rng = np.random.default_rng(rng)
    if normalize:
        scale = np.sqrt(1.0/min(n_rows, n_cols))
        S = rng.normal(0.0, scale, (n_rows, n_cols))
        # if more cols than rows (typical of embeddings),
        # want E[S.T @ S] = I. If more rows than cols
        # (typical of test matrices in low-rank factorizations),
        # want E[S @ S.T] = I
    else:
        S = rng.standard_normal((n_rows, n_cols))
    return S


def sjlt_operator(n_rows, n_cols, rng=None, vec_nnz=8):
    """

    Parameters
    ----------
    rng
    n_rows : int
        number of rows of embedding operator
    n_cols : int
        number of columns of embedding operator
    vec_nnz : int
        number of nonzeros in each column (if n_cols > n_rows) or each row (if n_rows >= n_cols)

    Returns
    -------
    S : SciPy sparse matrix
    """
    rng = np.random.default_rng(rng)
    if n_cols >= n_rows:
        vec_nnz = min(n_cols, vec_nnz)
        # column and row indices
        row_vecs = []
        for i in range(n_cols):
            rows = rng.choice(n_rows, vec_nnz, replace=False)
            row_vecs.append(rows)
        rows = np.concatenate(row_vecs)
        cols = np.repeat(np.arange(n_cols), vec_nnz)
        # values for each row and col
        vals = np.ones(n_cols * vec_nnz)
        vals[rng.random(n_cols * vec_nnz) <= 0.5] = -1
        vals /= np.sqrt(vec_nnz)
        # wrap up
        S = spar.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
        S = S.tocsc()
    else:
        #TODO: make this more efficient. (Form S directly, avoid converting
        #   from CSC to CSR.)
        S = sjlt_operator(n_cols, n_rows, rng, vec_nnz)
        S = (S.T).tocsr()
    return S


def sparse_sign_operator(n_rows, n_cols, rng=None, density=0.05):
    # get row indices and col indices
    rng = np.random.default_rng(rng)
    nonzero_idxs = rng.random(n_rows * n_cols) < density
    attempt = 0
    while np.all(~nonzero_idxs):
        if attempt == 10:
            raise RuntimeError('Density too low.')
        nonzero_idxs = rng.random(n_rows * n_cols) < density
        attempt += 1
    nonzero_idxs = np.where(nonzero_idxs)[0]
    rows, cols = np.unravel_index(nonzero_idxs, (n_rows, n_cols))
    # get values for each row and col index
    nnz = rows.size
    vals = np.ones(nnz)
    vals[rng.random(vals.size) < 0.5] = -1
    vals /= np.sqrt(min(n_rows, n_cols) * density)
    # Wrap up
    S = spar.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))
    S = S.tocsr()
    return S


def srct_data(n_rows, n_cols, rng):
    big_dim = max(n_rows, n_cols)
    small_dim = min(n_rows, n_cols)
    r = rng.choice(big_dim, size=small_dim, replace=False)
    e = rng.random(big_dim)
    e[e > 0.5] = 1.0
    e[e != 1] = -1.0
    e *= np.sqrt(big_dim / small_dim)
    perm = rng.permutation(big_dim)
    return r, e, perm


def apply_srct(r, e, mat, perm=None):
    """
    Apply a subsampled randomized cosine transform (SRCT) to the columns
    of the ndarray mat. The transform is defined by data (r, e).

    Parameters
    ----------
    r : ndarray
        The random restriction used in the SRCT. The entries of "r" must
        be unique integers between 0 and mat.shape[0] (exclusive).
    e : ndarray
        The vector of signs used in the SRCT; e.size == mat.shape[0].
    mat : ndarray
        The operand for the embedding. If mat.ndim == 1, then simply apply
        the SRCT to mat as a vector.
    perm : ndarray
        permutation of range(mat.shape[0]).

    Returns
    -------
    mat : ndarray
        The transformed input.
    """
    #TODO: check that dct performance isn't suffering from memory alignment issues.
    #
    #TODO: consider using SRCT in scipy.linalg._interpolative_backend -- that takes
    #   advantage of efficiency gain by subsampling.
    if mat.ndim > 1:
        if perm is not None:
            mat = mat[perm, :]
        mat = mat * e[:, None]
        mat = dct(mat, axis=0, norm='ortho')
        mat = mat[r, :]
    else:
        if perm is not None:
            mat = mat[perm]
        mat = mat * e
        mat = dct(mat, norm='ortho')
        mat = mat[r]
    return mat
