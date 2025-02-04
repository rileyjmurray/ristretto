import numpy as np
import warnings
import scipy.linalg as la


def orth(S):
    return la.qr(S, mode='economic')[0]


def lu_stabilize(S):
    L = la.lu(S, permute_l=True)[0]
    return L


def lupt(M):
    """Factor M = L @ U @ P.T. Equivalently, M @ P = L @ U."""
    P, L, U = la.lu(M)
    return U.T, L.T, P


def fixed_rank_warning(eager, tol, early_stop_possible=False):
    if not eager and tol < np.inf:
        if early_stop_possible:  # Might want to raise an exception.
            msg = """
            This implementation can only control error tolerance as a means 
            of early termination. No guarantee can be made that we will come 
            close to the requested tolerance.
            """
        else:  # Should probably raise an exception.
            msg = """
            This implementation has no control over error tolerance.
            """
        warnings.warn(msg)
