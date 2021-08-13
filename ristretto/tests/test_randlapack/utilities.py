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
    """
    min{ || [1, x] @ [a; b] - log(y) ||_2 : a, b are real numbers }

    Return the coefficients [a, b] and the R-squared of the fit.

    When calling this function to determine qualitative convergence rates,
    getting a < 0 by a decent margin (e.g., a < -0.5) and a large R-squared
    (e.g., r2 >= 0.8) gives a good indication of linear convergence.
    """
    assert x.size == y.size
    x = x.ravel()
    if np.any(y <= 0):
        warnings.warn('Dropping samples "i" where y[i] <= 0.')
        x = x[y > 0]
        y = y[y > 0]
    logy = np.log(y).ravel()
    mat = np.column_stack([np.ones(x.size), x])
    fit = la.lstsq(mat, logy)[0]
    ss_tot = np.sum((logy - np.mean(logy))**2)
    ss_res = np.sum((logy - mat @ fit)**2)
    r2 = 1 - ss_res/ss_tot
    return fit, r2


def loglog_fit(x, y):
    """
    min{ || [1, log(x)] @ [a; b] - log(y)||_2 : a, b are real numbers }

    Return coefficients (a, b) and the R-squared of the fit.

    When calling this function to determine qualitative convergence rates,
    getting a < 0 by a decent margin and a large R-squared gives a good
    indication of SUBLINEAR convergence. In that case, you're looking at
    errors which decay at a rate O(t^(-|a|)).
    """
    assert x.size == y.size
    x = x.ravel()
    if np.any(y <= 0):
        warnings.warn('Dropping samples "i" where y[i] <= 0.')
        x = x[y > 0]
        y = y[y > 0]
    if np.any(x <= 0):
        raise ValueError('Input x must be positive.')
    logy = np.log(y).ravel()
    mat = np.column_stack([np.ones(x.size), np.log(x)])
    fit = la.lstsq(mat, logy)[0]
    ss_tot = np.sum((logy - np.mean(logy))**2)
    ss_res = np.sum((logy - mat @ fit)**2)
    r2 = 1 - ss_res/ss_tot
    return fit, r2
