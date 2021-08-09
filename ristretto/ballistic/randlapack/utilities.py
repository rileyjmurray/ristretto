import numpy as np
import warnings
import scipy.linalg as la


def orth(S):
    return la.qr(S, mode='economic')[0]


def fixed_rank_warning(eager, tol, early_stop_possible=False):
    if not eager and tol < np.inf:
        if early_stop_possible:
            msg = """
            This implementation can only control error tolerance as a means 
            of early termination. No guarantee can be made that we will come 
            close to the requested tolerance.
            """
        else:
            msg = """
            This implementation has no control over error tolerance.
            """
        warnings.warn(msg)
