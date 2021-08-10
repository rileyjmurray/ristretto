"""
Routines for the orthogonal rangefinder problem:

    Given an m-by-n input matrix A, find a matrix Q with k << min(m, n)
    orthogonal columns where the operator norm || A - Q Q' A ||_2 is small.

    This problem can be considered where k is given or when a tolerance "tol"
    is given and we require || A - Q Q' A ||_2 <= tol.
"""
import warnings
import numpy as np
import scipy.linalg as la
import ristretto.ballistic.randlapack.utilities as util
from ristretto.ballistic.rblas.sketching import gaussian_operator
from ristretto.ballistic.randlapack.comps.powering import PoweredSketchOp

###############################################################################
#       Classic implementations, exposing fewest possible parameters.
###############################################################################


def power_rangefinder(A, k, num_pass,
                      sketch_op_gen=None, rng=None,
                      stabilizer=None, pps=1):
    """
    When building the matrix Q we are allowed to access A or A.T a total
    of num_pass times. See the function "power_rangefinder_sketch_op" for the
    meaning of the parameter "pps".

    sketch_op_gen is a function handle that accepts two positive integer
    arguments and one argument of various possible types (None, int,
    np.random.SeedSequence, np.random.BitGenerator, np.random.Generator) to
    control the random number generation process. The value
        mat = sketch_op_gen(k1, k2, rng)
    should be a k1-by-k2 numpy ndarray. If sketch_op_gen is not provided,
    we define it so that it generates a matrix with iid standard normal entries.
    """
    rng = np.random.default_rng(rng)
    if sketch_op_gen is None:
        sketch_op_gen = gaussian_operator
    if stabilizer is None:
        stabilizer = util.orth
    rf = RF1(num_pass, pps, stabilizer, sketch_op_gen)
    Q = rf.exec(A, k, 0.0, rng)
    return Q

###############################################################################
#       Object oriented interfaces
###############################################################################


class RangeFinder:

    def exec(self, A, k, tol, rng):
        """
        Return a matrix Q with orthonormal columns, where Range(Q) is
        "reasonably" closely aligned with the top dim(Range(Q)) left
        singular vectors of A.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix whose range is to be approximated.

        k : int
            Target for the number of columns in Q: 0 < k <= min(A.shape).
            Typically, k << min(A.shape). Conformant implementations ensure
            Q has at most k columns. For certain implementations it's
            reasonable to choose k as large as k = min(A.shape), in which
            case the implementation returns only once a specified error
            tolerance has been met.

        tol : float
            Target for the error  ||A - Q Q' A||: 0 <= tol < np.inf. Only
            certain implementations are able to control approximation error.
            Those implementations may return a matrix Q with fewer than k
            columns if ||A - Q Q' A|| <= tol. Assuming k < rank(A) and that the
            implementation can compute ||A - Q Q' A|| accurately, setting
            tol=0 means the implementation will return Q with exactly k columns.

            Implementations that cannot control error should raise a warning
            if tol > 0. The rationale for this behavior is that setting
            tol > 0 indicates an intention on the user's part that approximation
            error play a role in the stopping criteria.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Notes
        -----
        Here is the simplest possible implementation:

            rng = np.random.default_rng(rng)
            S = rng.standard_normal((A.shape[1], k))
            Y = A @ S
            Q = la.qr(Y, mode='economic')[0]

        """
        raise NotImplementedError()


class RF1(RangeFinder):

    def __init__(self, num_pass, pps, stabilizer, sketch_op_gen):
        if sketch_op_gen is None:
            sketch_op_gen = gaussian_operator
        if stabilizer is None:
            stabilizer = util.orth
        self.num_pass = num_pass
        self.sketch_op_gen = sketch_op_gen
        self.stabilizer = stabilizer
        self.pps = pps

    def exec(self, A, k, tol, rng):
        assert k > 0
        assert k <= min(A.shape)
        assert tol < np.inf
        if tol > 0:
            msg = """
            This RangeFinder implementation cannot directly control
            approximation error. Parameter "tol" is being ignored.
            """
            warnings.warn(msg)
        rng = np.random.default_rng(rng)
        S = PoweredSketchOp(self.num_pass, self.pps,  self.stabilizer,
                            self.sketch_op_gen).exec(A, k, rng)
        Y = A @ S
        Q = la.qr(Y, mode='economic')[0]
        return Q
