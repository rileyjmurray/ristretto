"""
Routines for the orthogonal rangefinder problem:

    Given an m-by-n input matrix A, find a matrix Q with k << min(m, n)
    orthogonal columns where the operator norm || A - Q Q' A ||_2 is small.

    This problem can be considered where k is given or when a tolerance "tol"
    is given and we require || A - Q Q' A ||_2 <= tol.
"""
import numpy as np
import scipy.linalg as la
import ristretto.ballistic.randlapack.utilities as util
from ristretto.ballistic.rblas.sketching import gaussian_operator
from ristretto.ballistic.randlapack.comps.powering import PoweredSketchOp


class RangeFinder:

    def exec(self, A, k, tol, eager, rng):
        """
        Return a matrix Q with orthonormal columns, where Range(Q) is
        "reasonably" closely aligned with the top dim(Range(Q)) left
        singular vectors of A.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix whose range is to be approximated.
        k : int
            Target for the number of columns in Q.
        tol : float
            Target for the error || A - Q Q' A ||.
        eager : bool
            If True, then terminate as soon as soon as possible after Q has
            k columns OR the error drops below tol. If False, then terminate
            as soon as possible after Q has at least k columns AND the error
            drops below tol. The meaning of the phrase "as soon as possible"
            is implementation dependent. Different implementations might not
            be able to control error tolerance and might ignore this argument.
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
    Q = rf.exec(A, k, np.inf, True, rng)
    return Q


class RF1(RangeFinder):

    def __init__(self, num_pass, pps, stabilizer, sketch_op_gen):
        if sketch_op_gen is None:
            sketch_op_gen = gaussian_operator
        if stabilizer is None:
            stabilizer = orth
        self.num_pass = num_pass
        self.sketch_op_gen = sketch_op_gen
        self.stabilizer = stabilizer
        self.pps = pps

    def exec(self, A, k, tol, eager, rng):
        util.fixed_rank_warning(eager, tol, early_stop_possible=False)
        rng = np.random.default_rng(rng)
        S = PoweredSketchOp(self.num_pass, self.pps,  self.stabilizer,
                            self.sketch_op_gen).exec(A, k, rng)
        Y = A @ S
        Q = la.qr(Y, mode='economic')[0]
        return Q
