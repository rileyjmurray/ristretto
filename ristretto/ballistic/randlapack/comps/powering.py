import numpy as np
import ristretto.ballistic.rblas.sketching as rist_sk
import ristretto.ballistic.randlapack.utilities as rist_util


def powered_range_sketch_op(num_pass, A, k, rng):
    """
    Return an n-by-k matrix S for use in sketching the rows of the m-by-n
    matrix A. (I.e., for computing a sketch Y = A @ S.) The qualitative goal
    is that the range of S should be well-aligned with the top-k right
    singular vectors of A.

    This function works by taking "num_pass" steps of a power method that
    starts with a random Gaussian matrix, and then makes alternating
    applications of A and A.T. We stabilize the power method with a QR
    factorization.

    Setting num_pass = 0 is a valid option.
    """
    assert num_pass >= 0
    assert k >= 1
    assert k <= min(A.shape)
    S = PRSO1(sketch_op_gen=rist_sk.gaussian_operator,
              num_pass=num_pass,
              stabilizer=rist_util.orth,
              passes_per_stab=1).exec(A, k, rng)
    return S


class RowSketchingOperator:
    """
    Given a matrix A and a positive integer k, generates a matrix S with k
    columns, for later use in sketching Y = A @ S. By virtue of taking
    linear combinations of the columns, the matrix S is essentially sketching
    the rows of A.
    """

    def exec(self, A, k, rng):
        """
        Return a matrix S where range(S) is "reasonably" well
        aligned with the span of the top k right singular vectors
        of A. Do this while optionally incorporating information
        about A, e.g., by subspace powering. It's possible that
        we construct the matrix S without accessing A, but in that
        situation the meaning of "reasonable" is very weak.

        Parameters
        ----------
        A : Union[ndarray, spmatrix, LinearOperator]
            Data matrix we'll sketch later with Y = A @ S.

        k : int
            Number of columns of S.

        rng : Union[None, int, SeedSequence, BitGenerator, Generator]
            Determines the numpy Generator object that manages any and all
            randomness in this function call.

        Notes
        -----
        The simplest possible implementation of this function is

            S = np.random.standard_normal((A.shape[1], k))
            return S

        That is -- there is no requirement that implementations actually
        read the entries of A.
        """
        raise NotImplementedError()


class PRSO1(RowSketchingOperator):
    """
    Powered Row Sketching Operator

    PRSO1 objects are used to create n-by-k matrices S for use in sketching
    the rows of an m-by-n matrix A. The qualitative goal is that the range
    of S should be well-aligned with the top-k right singular vectors of A.

    PRSO objects work by applying a power method that starts with an initial
    random matrix with k columns, and then makes alternating applications of
    A and A.T. The tuning parameters in this procedure are:

        How we generate the initial random matrix.
        The number of passes over A (or A.T).
        How we stabilize the power method. E.g., QR or LU factorization.
        How often we stabilize the power method.
    """

    def __init__(self, sketch_op_gen, num_pass, stabilizer, passes_per_stab):
        self.sketch_op_gen = sketch_op_gen
        self.num_pass = num_pass
        self.stabilizer = stabilizer
        self.passes_per_stab = passes_per_stab

    def exec(self, A, k, rng):
        """
        Use self.num_pass passes over the matrix A to generate a matrix S
        where range(S) is (hopefully) closely aligned with the span of A's
        top right singular vectors. This is useful for estimating the span of
        the top *left* singular vectors of A, by evaluating Y = A @ S.

        We accomplish this roughly as follows:

        if self.num_pass is even
            S = (A' A)^(self.num_pass/2) self.sketch_op_gen(n, k, rng)
        if self.num_pass is odd
            S = (A' A)^((self.num_pass-1)/2) A' self.sketch_op_gen(m, k, rng)

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
        """
        rng = np.random.default_rng(rng)
        external_num_pass = self.num_pass + 1
        if external_num_pass % 2 == 1:
            S = self.sketch_op_gen(A.shape[1], k, rng)
            passes_done = 0
            q = (external_num_pass - 1) // 2
        else:
            S = A.T @ self.sketch_op_gen(A.shape[0], k, rng)
            passes_done = 1
            if self.passes_per_stab == 1:
                S = self.stabilizer(S)
            q = (external_num_pass - 2) // 2
        # q is an even integer; need to compute
        #   S := (A' A)^q S
        # up to intermediate stabilization.
        while q > 0:
            S = A @ S
            passes_done += 1
            if passes_done % self.passes_per_stab == 0:
                S = self.stabilizer(S)
            S = A.T @ S
            passes_done += 1
            if passes_done % self.passes_per_stab == 0:
                S = self.stabilizer(S)
            q -= 1
        return S
