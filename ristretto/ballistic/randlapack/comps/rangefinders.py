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
from ristretto.ballistic.randlapack.comps.powering import powered_range_sketch_op


def power_rangefinder(A, k, num_pass, sketch_op_gen=None, gen=None, stabilizer=None, pps=1):
    """
    When building the matrix Q we are allowed to access A or A.T a total
    of num_pass times. See the function "power_rangefinder_sketch_op" for the
    meaning of the parameter "pps".

    sketch_op_gen is a function handle that accepts two positive integer arguments
    and one argument of various possible types (None, int, np.random.SeedSequence,
    np.random.BitGenerator, np.random.Generator) to control the random number
    generation process. The value
        mat = sketch_op_gen(k1, k2, gen)
    should be a k1-by-k2 numpy ndarray. If sketch_op_gen is not provided, we define
    it so that it generates a matrix with iid standard normal entries.
    """
    gen = np.random.default_rng(gen)

    if sketch_op_gen is None:
        sketch_op_gen = gaussian_operator

    if stabilizer is None:
        def stabilizer(mat):
            return la.qr(mat, mode='economic')[0]

    S = powered_range_sketch_op(A, k, num_pass, sketch_op_gen, stabilizer, pps, gen)
    Y = A @ S
    Q = la.qr(Y, mode='economic')[0]
    return Q
