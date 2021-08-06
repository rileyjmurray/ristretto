"""
Routines for the orthogonal rangefinder problem:

    Given an m-by-n input matrix A, find a matrix Q with k << min(m, n)
    orthogonal columns where the operator norm || A - Q Q' A ||_2 is small.

    This problem can be considered where k is given or when a tolerance "tol"
    is given and we require || A - Q Q' A ||_2 <= tol.
"""
import numpy as np
import scipy.linalg as la
from ristretto.ballistic.rblas.sketching import gaussian_operator
from ristretto.ballistic.randlapack.oo_comps.powering import PoweredSketchOp


def orth(S):
    return la.qr(S, mode='economic')


class FRRF:
    """
    Fixed rank rangefinder
    """

    def __call__(self, A, k, gen, **kwargs):
        """
        Find a matrix Q that has k orthonormal columns where || A - Q Q' A ||
        is "reasonably" close to the error || A - A_k || of the best rank-k
        approximation of A. The range of Q serves as an approximation for the
        the span of the top k left singular vectors of A.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix whose range is to be approximated.

        k : int
            Number of columns in Q.

        gen : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Notes
        -----
        Here is the simplest possible implementation:

            gen = np.random.default_rng(gen)
            S = gen.standard_normal((A.shape[1], k))
            Y = A @ S
            Q = la.qr(Y, mode='economic')[0]

        """
        raise NotImplementedError()


class PowerRangeFinder(FRRF):

    def __init__(self, num_pass, pps, stabilizer, sketch_op_gen):

        if sketch_op_gen is None:
            sketch_op_gen = gaussian_operator

        if stabilizer is None:
            stabilizer = orth

        self.num_pass = num_pass
        self.sketch_op_gen = sketch_op_gen
        self.stabilizer = stabilizer
        self.pps = pps

    def __call__(self, A, k, gen, **kwargs):
        gen = np.random.default_rng(gen)
        sk_op = PoweredSketchOp(self.num_pass, self.pps, self.stabilizer, self.sketch_op_gen)
        S = sk_op(A, k, gen)
        Y = A @ S
        Q = la.qr(Y, mode='economic')[0]
        return Q
