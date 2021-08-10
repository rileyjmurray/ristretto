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


def power_rangefinder(A, k, num_pass, rng):
    """
    Return a matrix Q with k orthonormal columns, where Range(Q) is
    an approximation for the span of A's top k left singular vectors.

    This implementation uses a Gaussian sketching matrix with k columns
    and requires a total of num_pass over A. We use QR to orthogonalize
    the output of every matrix multiply that involves A or A'.

    Parameters
    ----------
    A : Union[ndarray, spmatrix, LinearOperator]
        Data matrix whose range is to be approximated.

    k : int
        Q.shape[1] == k. We require 0 < k <= min(A.shape).

    num_pass : int
        Total number of passes over A.

    rng : Union[None, int, SeedSequence, BitGenerator, Generator]
        Determines the numpy Generator object that manages any and all
        randomness in this function call.

    Returns
    -------
    Q : ndarray
        Q.shape = (A.shape[0], k) has orthonormal columns.
    """

    rng = np.random.default_rng(rng)
    rf = RF1(num_pass, passes_per_stab=1, stabilizer=util.orth,
             sketch_op_gen=gaussian_operator)
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

    def __init__(self, num_pass, passes_per_stab, stabilizer, sketch_op_gen):
        if sketch_op_gen is None:
            sketch_op_gen = gaussian_operator
        if stabilizer is None:
            stabilizer = util.orth
        self.num_pass = num_pass
        self.sketch_op_gen = sketch_op_gen
        self.stabilizer = stabilizer
        self.passes_per_stab = passes_per_stab

    def exec(self, A, k, tol, rng):
        """
        Return a matrix Q with k orthonormal columns, where Range(Q) is
        an approximation for the span of A's top k left singular vectors.

        Suppose A is m-by-n. This implementation uses (self.num_pass - 1)
        passes over A to construct a sketching matrix S of dimensions (n, k).
        Once S is in hand, we form "Y = A S and" return the factor Q
        from a QR decomposition of Y.

        Here is how we would construct S in exact arithmetic:

        if num_pass is odd
            S = (A' A)^((self.num_pass-1)/2) self.sketch_op_gen(n, k, rng)
        if num_pass is even
            S = (A' A)^((self.num_pass-2)/2) A' self.sketch_op_gen(m, k, rng)

        The actual matrix S matches the matrices above only up to its range.
        The discrepancy is because forming the matrices above would result in
        loss of precision from successive applications of (A, A'). We mitigate
        this precision loss by the following procedure:

        After "self.passes_per_stab" applications of A or A', we replace the
        working matrix S (which might be of shape (n, k) or (m, k) at the time)
        by S = self.stabilizer(S). An implementation of "self.stabilizer" is
        valid as long as it returns a numerically well-behaved basis for the
        range of its argument. The most common choice of self.stabilizer is
        to return the factor Q from an (economic) QR factorization. An
        alternative choice is to return the factor L from an LU decomposition.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix whose range is to be approximated.

        k : int
            Q.shape[1] == k.

        tol : float
            Refer to the RangeFinder interface for the general meaning
            of this parameter. This implementation checks if top < np.inf,
            returns a warning if tol > 0, but otherwise ignores tol.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Returns
        -------
        Q : ndarray
            Q.shape = (A.shape[0], k) and has orthonormal columns.
        """
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
        S = PoweredSketchOp(self.num_pass, self.passes_per_stab,
                            self.stabilizer, self.sketch_op_gen).exec(A, k, rng)
        Y = A @ S
        Q = la.qr(Y, mode='economic')[0]
        return Q
