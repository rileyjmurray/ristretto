from typing import Union
import numpy as np
import scipy.linalg as la
import ristretto.ballistic.randlapack.utilities as util
import warnings


def rand_onb(n_rows, n_cols, rng):
    rng = np.random.default_rng(rng)
    Q = rng.standard_normal(size=(n_rows, n_cols))
    Q = util.orth(Q)
    return Q


def rand_low_rank(n_rows, n_cols, spectrum: Union[int, np.ndarray], rng):
    rng = np.random.default_rng(rng)
    if isinstance(spectrum, int):
        spectrum = rng.random(size=(spectrum,))
    spectrum = np.sort(spectrum)
    spectrum = spectrum[::-1]  # reverse
    spectrum /= spectrum[0]
    rank = spectrum.size
    U = rand_onb(n_rows, rank, rng)
    V = rand_onb(rank, n_cols, rng)
    M = (U * spectrum) @ V
    return M


def loglinear_fit(x, y):
    assert x.size == y.size
    x = x.ravel()
    if np.any(y <= 0):
        warnings.warn('Dropping samples "i" where y[i] == 0.')
        x = x[y > 0]
        y = y[y > 0]
    logy = np.log(y).ravel()
    mat = np.column_stack([np.ones(x.size), x])
    fit, residues = la.lstsq(mat, logy)[:2]
    ss_tot = np.sum((logy - np.mean(logy))**2)
    ss_res = np.sum((logy - mat @ fit)**2)
    r2 = 1 - ss_res/ss_tot
    return fit, r2
