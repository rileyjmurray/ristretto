import numpy as np
from scipy.sparse.linalg import LinearOperator


def sampling_operator(n_rows, n_cols, gen=None, indices=None):
    gen = np.random.default_rng(gen)
    if indices is None:
        pop_size = max(n_rows, n_cols)
        sample_size = min(n_rows, n_cols)
        indices = gen.choice(pop_size, sample_size, replace=False)
        indices.sort()
    else:
        assert indices.size == min(n_rows, n_cols)
        # should also have np.all(0 <= indices) and np.all(indices < max(n_rows, n_cols))
    if n_cols >= n_rows:
        def matvec(vec):
            return vec[indices]
        def matmat(mat):
            return mat[indices, :]
        def rmatvec(vec):
            out = np.zeros(n_cols)
            out[indices] = vec
            return out
        def rmatmat(mat):
            out = np.zeros(mat.shape[0], n_cols)
            out[:, indices] = mat
        S = LinearOperator(shape=(n_rows, n_cols),
                           matvec=matvec, matmat=matmat,
                           rmatvec=rmatvec, rmatmat=rmatmat)
    else:
        #TODO: form S directly.
        S = sampling_operator(n_cols, n_rows, indices)
        S = S.T
    return S
